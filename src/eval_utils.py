"""Helpers shared by the notebooks."""

from __future__ import annotations

import time
from pathlib import Path

import torch


def find_root():
    for p in (Path.cwd(), Path.cwd().parent):
        if (p / "src").is_dir() and (p / "notebooks").is_dir():
            return p
    return Path.cwd()


def model_stats(yolo):
    info = yolo.info(verbose=False)
    if isinstance(info, (tuple, list)) and len(info) >= 4:
        n_l, n_p, n_g, flops = info[:4]
    else:
        n_l = n_p = n_g = flops = None
    return {"layers": n_l, "params": n_p, "gradients": n_g, "gflops": flops}


def val_map(yolo, data):
    metrics = yolo.val(data=data, verbose=False)
    return {
        "map50": float(metrics.box.map50),
        "map": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def benchmark_ms(yolo, source, n=30, warmup=5):
    for _ in range(warmup):
        yolo.predict(source, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        yolo.predict(source, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


def coco128_train_images():
    candidates = [
        Path.cwd() / "datasets" / "coco128" / "images" / "train2017",
        find_root() / "datasets" / "coco128" / "images" / "train2017",
        Path.cwd().parent / "datasets" / "coco128" / "images" / "train2017",
    ]
    try:
        from ultralytics.data.utils import check_det_dataset
        data = check_det_dataset("coco128.yaml")
        path = Path(data["path"])
        candidates.insert(0, path / "images" / "train2017")
        if "train" in data:
            candidates.insert(0, Path(data["train"]))
    except Exception:
        pass
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]
