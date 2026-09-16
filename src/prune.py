"""Structured channel pruning using BN scale gamma (paper Sec. 3.1.2)."""

from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck, C2f, Conv, Detect, SPPF


def prune_conv(conv1, conv2, threshold=0.01, min_channels=8):
    """Prune conv1 output channels and matching conv2 input channels."""
    gamma = conv1.bn.weight.data.detach()
    beta = conv1.bn.bias.data.detach()
    keep_idxs = torch.tensor([], dtype=torch.long)
    local_threshold = threshold
    while len(keep_idxs) < min_channels:
        keep_idxs = torch.where(gamma.abs() >= local_threshold)[0]
        local_threshold *= 0.5

    n = len(keep_idxs)
    conv1.bn.weight.data = gamma[keep_idxs]
    conv1.bn.bias.data = beta[keep_idxs]
    conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idxs]
    conv1.bn.running_mean.data = conv1.bn.running_mean.data[keep_idxs]
    conv1.bn.num_features = n
    conv1.conv.weight.data = conv1.conv.weight.data[keep_idxs]
    conv1.conv.out_channels = n
    if conv1.conv.bias is not None:
        conv1.conv.bias.data = conv1.conv.bias.data[keep_idxs]

    if not isinstance(conv2, list):
        conv2 = [conv2]
    for item in conv2:
        if item is None:
            continue
        conv = item.conv if isinstance(item, Conv) else item
        conv.in_channels = n
        conv.weight.data = conv.weight.data[:, keep_idxs]


def prune_module(m1, m2, threshold=0.01, min_channels=8):
    """Prune the junction from module m1 (bottom conv) to m2 (top conv)."""
    if isinstance(m1, C2f):
        m1 = m1.cv2
    if not isinstance(m2, list):
        m2 = [m2]
    for i, item in enumerate(m2):
        if isinstance(item, (C2f, SPPF)):
            m2[i] = item.cv1
    prune_conv(m1, m2, threshold, min_channels)


def bn_gamma_threshold(model, keep_ratio=0.8):
    """Global threshold: keep the top `keep_ratio` BN gamma magnitudes."""
    ws = []
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            ws.append(m.weight.abs().detach())
    ws = torch.cat(ws)
    idx = min(int(len(ws) * keep_ratio), len(ws) - 1)
    return torch.sort(ws, descending=True)[0][idx]


def prune_yolov8(weights, save_path, keep_ratio=0.8, min_channels=8):
    """Prune YOLOv8n at the junctions listed in the paper.

    Skipped: stem layers 0-2; layers 4/6/9 (Detect concat); C2f split / FPN upsample.
    Pruned: backbone 3->4, 5->6, 7->8, 8->9; head 15/18/21; Bottleneck and Detect internals.
    """
    yolo = YOLO(str(weights))
    model = yolo.model
    threshold = bn_gamma_threshold(model, keep_ratio)
    print(f"BN gamma threshold @ keep_ratio={keep_ratio}: {threshold.item():.6f}")

    for m in model.modules():
        if isinstance(m, Bottleneck):
            prune_conv(m.cv1, m.cv2, threshold, min_channels)

    seq = model.model
    for i in range(3, 9):
        if i in (4, 6, 9):
            continue
        prune_module(seq[i], seq[i + 1], threshold, min_channels)

    detect = seq[-1]
    assert isinstance(detect, Detect)
    last_inputs = [seq[15], seq[18], seq[21]]
    colasts = [seq[16], seq[19], None]
    for last_input, colast, cv2, cv3 in zip(last_inputs, colasts, detect.cv2, detect.cv3):
        prune_module(last_input, [colast, cv2[0], cv3[0]], threshold, min_channels)
        prune_module(cv2[0], cv2[1], threshold, min_channels)
        prune_module(cv2[1], cv2[2], threshold, min_channels)
        prune_module(cv3[0], cv3[1], threshold, min_channels)
        prune_module(cv3[1], cv3[2], threshold, min_channels)

    for p in yolo.model.parameters():
        p.requires_grad = True

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    yolo.save(str(save_path))
    print(f"saved pruned model -> {save_path}")
    return yolo
