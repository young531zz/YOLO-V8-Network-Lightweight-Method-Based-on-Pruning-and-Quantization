# 基于剪枝与量化的 YOLOv8 轻量化方法

论文 **YOLO V8 Network Lightweight Method Based on Pruning and Quantization**（Cheng, 2024）的代码复现。

> Cheng, Y. (2024). YOLO V8 Network Lightweight Method Based on Pruning and Quantization. *Mathematical Modeling and Algorithm Application*, 3(2), 44–51. ISSN: 3006-0842

本仓库将原始实验脚本整理为两个可运行 notebook，分别对应论文 Figure 2（剪枝）与 Figure 3（量化）。

---

## 方法

完整流水线：**稀疏训练 → BN 通道剪枝 → 重训练 → TensorRT INT8（min-max）**

### 1. 稀疏训练

对 BN 层尺度 \(\gamma\) 与偏置 \(\beta\) 施加 L1 约束（论文公式 5–9）：

\[
L = \sum \ell(f(x), y) + \lambda_1 \sum |\gamma| + \lambda_2 \sum |\beta|
\]

\[
\lambda_1 = 0.01 \times (1 - 0.9 \cdot e / n_e)
\]

实现上通过 Ultralytics 的 `optimizer_step` 补丁完成，**不修改** ultralytics 源码。

### 2. 结构化通道剪枝

以 BN 的 \(|\gamma|\) 作为通道重要性。\(\gamma \approx 0\) 的通道及其上下游卷积核会被删除。

论文指定的可剪位置（YOLOv8n）：

- Backbone：`Conv(3)→C2f(4)`，`Conv(5)→C2f(6)`，`Conv(7)→C2f(8)`，`C2f(8)→SPPF(9)`
- Head：`C2f(15/18/21)` 到后续 Conv / Detect
- 模块内部：C2f Bottleneck 的 `cv1→cv2`，Detect 的 `cv2/cv3` 卷积链

不剪：前三层；第 4/6/9 层（与 Detect 拼接有关）；C2f split、FPN Upsample/Concat。

剪枝后每个卷积至少保留 8 个通道，便于硬件加速。

### 3. INT8 量化

剪枝后将 float32 权重与激活量化为 int8。论文使用 min-max（公式 10–11）：

\[
q = \frac{x - \min}{\max - \min} \times 255, \quad
x = \frac{q \times (\max - \min)}{255} + \min
\]

部署框架为 NVIDIA TensorRT（`IInt8MinMaxCalibrator`）。

---

## 论文结果

实验设置：YOLOv8n，NVIDIA GeForce RTX 3050，COCO。论文中 Acc 为检测精度指标。

**表 1 — 剪枝**

| | GFLOPs | 参数量 | Acc |
|---|---:|---:|---:|
| 剪枝前 | 8.7 | 3,151,904 | 0.771 |
| 剪枝后 | 5.8 | 2,536,321 | 0.722 |
| 重训练 | — | — | 0.754 |

**表 2 — TensorRT INT8**

| | Engine 体积 | 速度 | Acc |
|---|---:|---:|---:|
| 量化前 | 20.1 MB | 53.5 ms | 0.754 |
| 量化后 | 7.2 MB | 205 ms | 0.722 |

剪枝 + 量化后压缩率约 **64.2%**，精度约下降 **4.2%**。

> 原始代码在 coco128 上做了流程验证；论文表格来自 COCO 实验。用 coco128 复现时，数值会与表格不同。

---

## 仓库结构

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
├── weights/                       # 导出模型（不纳入版本库）
├── paper.pdf
├── requirements.txt
└── README.md
```

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

## 复现步骤

在 `notebooks/` 下打开 Jupyter，或从仓库根目录启动：

```bash
jupyter notebook notebooks/01_sparse_prune.ipynb
```

**Notebook 01**

1. 对 `yolov8n.pt` 做 BN L1 稀疏训练
2. 按 \(\gamma\) 阈值剪通道，保存 `weights/prune.pt`
3. 重训练得到 `weights/retrain.pt`
4. 对比参数量 / GFLOPs / mAP（对应表 1）

**Notebook 02**（依赖 01 的 `retrain.pt`）

1. 导出 ONNX
2. 使用 Ultralytics TensorRT INT8，或论文同款 min-max 标定
3. 对比 engine 体积、时延、mAP（对应表 2）

快速跑通流程：保持 notebook 开头的 `QUICK_DEMO = True`（少量 epoch）。对齐论文设置：改为 `QUICK_DEMO = False`。

---

## 原始代码对应关系

| 原始文件 | 作用 |
|---|---|
| `ultralytics/.../trainer.py`（被注释的 L1） | 稀疏训练，现为 `src/sparsity.py` |
| `prune.py` | BN 通道剪枝，现为 `src/prune.py` |
| `prune_train.py` / `compare.py` | 重训练与表 1 对比，现为 notebook 01 |
| `int8.py` / `trt_int8_quantization_yolo8.py` | TensorRT INT8，现为 notebook 02 |
| `yolov8-pytorch_quantization-main/` | PTQ/QAT 额外实现，**不属于**论文主流程 |

本仓库不包含完整 Ultralytics 源码、训练日志或权重文件。

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
