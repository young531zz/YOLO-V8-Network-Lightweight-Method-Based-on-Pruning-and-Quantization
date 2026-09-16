"""TensorRT INT8 export with min-max calibration (paper Sec. 3.2)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2 as cv
import numpy as np


def export_onnx(pt_path, imgsz=640):
    from ultralytics import YOLO

    pt_path = Path(pt_path).resolve()
    yolo = YOLO(str(pt_path))
    out = yolo.export(format="onnx", imgsz=imgsz, opset=13, simplify=True)
    onnx_path = Path(out) if out else pt_path.with_suffix(".onnx")
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    return onnx_path, yolo


def iter_calib_batches(image_dir, input_name, shape, n_images=64):
    """Yield preprocessed NCHW float32 batches in [0, 1] (paper Eqs. 10-11)."""
    image_dir = Path(image_dir)
    paths = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    n_images = min(n_images, len(paths))
    batch_size, _, height, width = shape
    batch = np.zeros(shape, dtype=np.float32)
    filled = 0
    for i, path in enumerate(paths[:n_images]):
        bgr = cv.imread(str(path))
        if bgr is None:
            continue
        rgb = cv.resize(bgr, (width, height))[:, :, ::-1]
        batch[filled] = (rgb / 255.0).transpose(2, 0, 1)
        filled += 1
        if filled == batch_size:
            yield {input_name: batch}
            batch = np.zeros_like(batch)
            filled = 0
    if filled:
        yield {input_name: batch}


def yolo_metadata(yolo, imgsz=640):
    names = yolo.names if isinstance(yolo.names, dict) else {i: n for i, n in enumerate(yolo.names)}
    return {
        "description": "YOLOv8 pruned INT8 TensorRT engine",
        "author": "Ultralytics",
        "license": "AGPL-3.0 https://ultralytics.com/license",
        "date": datetime.now().isoformat(),
        "stride": 32,
        "task": "detect",
        "batch": 1,
        "imgsz": [imgsz, imgsz],
        "names": {str(k): v for k, v in names.items()},
    }


def build_int8_engine(
    onnx_path,
    yolo,
    calib_dir,
    engine_path=None,
    n_images=64,
    imgsz=640,
    calibration_method="min-max",
):
    """ONNX -> TensorRT INT8 with IInt8MinMaxCalibrator (paper default)."""
    import onnxruntime
    import tensorrt as trt
    from polygraphy.backend.trt import Calibrator, CreateConfig, EngineFromNetwork, NetworkFromOnnxPath

    onnx_path = Path(onnx_path).resolve()
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    shape = session.get_inputs()[0].shape
    if any(not isinstance(d, int) for d in shape):
        shape = (1, 3, imgsz, imgsz)

    cache_file = onnx_path.with_name(onnx_path.stem + "_int8.cache")
    if cache_file.exists():
        cache_file.unlink()

    cfg = CreateConfig(builder_optimization_level=5)
    cfg.int8 = True
    calib_cls = trt.IInt8MinMaxCalibrator if calibration_method == "min-max" else trt.IInt8EntropyCalibrator2
    cfg.calibrator = Calibrator(
        BaseClass=calib_cls,
        data_loader=iter_calib_batches(calib_dir, input_name, shape, n_images),
        cache=str(cache_file),
    )

    if engine_path is None:
        engine_path = onnx_path.with_name(f"{onnx_path.stem}_int8_{calibration_method}.engine")
    engine_path = Path(engine_path)

    build_engine = EngineFromNetwork(NetworkFromOnnxPath(str(onnx_path)), config=cfg)
    with build_engine() as engine, open(engine_path, "wb") as f:
        meta = json.dumps(yolo_metadata(yolo, imgsz=imgsz))
        f.write(len(meta).to_bytes(4, byteorder="little", signed=True))
        f.write(meta.encode())
        f.write(engine.serialize())
    print(f"saved engine -> {engine_path}")
    return engine_path
