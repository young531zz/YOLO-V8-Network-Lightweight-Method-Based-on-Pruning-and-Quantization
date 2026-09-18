# 基于剪枝与量化的 YOLOv8 轻量化方法

论文 *YOLO V8 Network Lightweight Method Based on Pruning and Quantization*（Cheng, 2024）的代码复现。

> Cheng, Y. (2024). YOLO V8 Network Lightweight Method Based on Pruning and Quantization. *Mathematical Modeling and Algorithm Application*, 3(2), 44-51. ISSN: 3006-0842

本仓库将原始实验脚本整理为两个可运行 notebook，分别对应论文 Figure 2（剪枝）与 Figure 3（量化）。

**请勿上传权重。** `*.pt` / `*.onnx` / `*.engine` 以及训练产物都不纳入版本库。`yolov8n.pt` 在运行 notebook 时由 Ultralytics 自动下载。

---

## 方法

完整流水线：**稀疏训练 → BN 通道剪枝 → 重训练 → TensorRT INT8（min-max）**

### 1. 稀疏训练

对 BN 层尺度 γ 与偏置 β 施加 L1 约束（论文公式 5–9）：

```
L  = Σ ℓ(f(x), y) + λ1 · Σ |γ| + λ2 · Σ |β|
λ1 = 0.01 × (1 - 0.9 × e / ne)
```

其中 e 为当前 epoch，ne 为总 epoch 数。实现上通过 Ultralytics 的 `optimizer_step` 补丁完成，不修改 ultralytics 源码。

### 2. 结构化通道剪枝

以 BN 的 |γ| 作为通道重要性。γ 接近 0 的通道及其上下游卷积核会被删除。

论文指定的可剪位置（YOLOv8n）：

- Backbone：`Conv(3)→C2f(4)`，`Conv(5)→C2f(6)`，`Conv(7)→C2f(8)`，`C2f(8)→SPPF(9)`
- Head：`C2f(15/18/21)` 到后续 Conv / Detect
- 模块内部：C2f Bottleneck 的 `cv1→cv2`，Detect 的 `cv2/cv3` 卷积链

不剪：前三层；第 4/6/9 层（与 Detect 拼接有关）；C2f split、FPN Upsample/Concat。

剪枝后每个卷积至少保留 8 个通道，便于硬件加速。

### 3. INT8 量化

剪枝后将 float32 权重与激活量化为 int8。论文使用 min-max（公式 10–11）：

```
q = (x - min) / (max - min) × 255
x = q × (max - min) / 255 + min
```

部署框架为 NVIDIA TensorRT（`IInt8MinMaxCalibrator`）。

---

## 论文结果

实验设置：YOLOv8n，NVIDIA GeForce RTX 3050，COCO。论文中 Acc 为检测精度指标。

**表 1 — 剪枝**

|        | GFLOPs | 参数量      | Acc   |
|--------|--------:|------------:|------:|
| 剪枝前 | 8.7     | 3,151,904   | 0.771 |
| 剪枝后 | 5.8     | 2,536,321   | 0.722 |
| 重训练 | —       | —           | 0.754 |

**表 2 — TensorRT INT8**

|        | Engine 体积 | 速度    | Acc   |
|--------|------------:|--------:|------:|
| 量化前 | 20.1 MB     | 205 ms | 0.754 |
| 量化后 | 7.2 MB      | 53.5 ms  | 0.722 |

剪枝 + 量化后压缩率约 **64.2%**，精度约下降 **4.2%**。

> 原始代码在 coco128 上做了流程验证；论文表格来自 COCO 实验。用 coco128 复现时，数值会与表格不同。

---

## 仓库结构

上传 GitHub 时只需下面这些文件（合计约 1–2 MB）。不要把 `weights/`、`runs/`、`datasets/` 或任何 `.pt/.onnx/.engine` 一并上传。

```
yolov8-prune-quant/
├── notebooks/
│   ├── 01_sparse_prune.ipynb      # 稀疏训练 → 剪枝 → 重训练
│   └── 02_tensorrt_int8.ipynb     # TensorRT INT8
├── src/
│   ├── sparsity.py                # BN L1 稀疏训练
│   ├── prune.py                   # 通道剪枝
│   ├── int8_trt.py                # min-max TensorRT INT8
│   └── eval_utils.py
├── paper.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

运行后会在本地生成 `weights/`（如 `prune.pt`、`retrain.pt`）。

---

## 环境

- Python ≥ 3.8
- PyTorch（建议使用 GPU）
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) ≥ 8.0.186
- Notebook 02 需要 NVIDIA GPU + TensorRT + Polygraphy（可选）

```bash
pip install -r requirements.txt
```

数据集默认 `coco128.yaml`（Ultralytics 会自动下载，约 7 MB）。若要接近论文表格，请换完整 COCO：

```python
DATA = "coco.yaml"   # 需自行准备 COCO 2017
```

---

## 复现

从仓库根目录启动：

```bash
jupyter notebook notebooks/01_sparse_prune.ipynb
```

**Notebook 01**

1. 对 `yolov8n.pt` 做 BN L1 稀疏训练
2. 按 γ 阈值剪通道，保存到本地 `weights/prune.pt`
3. 重训练得到本地 `weights/retrain.pt`
4. 对比参数量 / GFLOPs / mAP（对应表 1）

**Notebook 02**（依赖 01 在本地生成的 `retrain.pt`）

1. 导出 ONNX
2. 使用 Ultralytics TensorRT INT8，或论文同款 min-max 标定
3. 对比 engine 体积、时延、mAP（对应表 2）



---


---

## 引用

```bibtex
@article{cheng2024yolov8,
  title   = {YOLO V8 Network Lightweight Method Based on Pruning and Quantization},
  author  = {Cheng, Yiyang},
  journal = {Mathematical Modeling and Algorithm Application},
  volume  = {3},
  number  = {2},
  pages   = {44--51},
  year    = {2024},
  issn    = {3006-0842}
}
```

YOLOv8 / Ultralytics 遵循 AGPL-3.0。本仓库中的剪枝与量化脚本用于论文复现，与 Ultralytics 联用时请遵守其许可证。
