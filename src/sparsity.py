"""BN-channel L1 sparsity training (paper Eqs. 5-9)."""

from __future__ import annotations

import torch
from torch import nn


class BNSparsity:
    """L1 penalty on BN scale gamma and bias beta.

    Eq. 9:  lambda1 = 0.01 * (1 - 0.9 * e / ne)
    Eqs. 7-8: extra gradient = lambda * sign(param)
    """

    def __init__(self, epochs=50, lambda1_0=0.01, lambda2=0.01, decay=0.9):
        self.epochs = max(int(epochs), 1)
        self.lambda1_0 = lambda1_0
        self.lambda2 = lambda2
        self.decay = decay
        self._orig_optimizer_step = None

    def lambda1(self, epoch):
        return self.lambda1_0 * (1.0 - self.decay * epoch / self.epochs)

    def add_bn_l1_(self, model, epoch):
        lam1 = self.lambda1(epoch)
        for m in model.modules():
            if not isinstance(m, nn.BatchNorm2d):
                continue
            if m.weight.grad is not None:
                m.weight.grad.data.add_(lam1 * torch.sign(m.weight.data))
            if m.bias is not None and m.bias.grad is not None:
                m.bias.grad.data.add_(self.lambda2 * torch.sign(m.bias.data))

    def register_yolo_callbacks(self, yolo):
        """Patch AMP optimizer_step so L1 is applied on unscaled gradients."""
        from ultralytics.engine.trainer import BaseTrainer
        from ultralytics.utils.torch_utils import de_parallel

        if self._orig_optimizer_step is None:
            self._orig_optimizer_step = BaseTrainer.optimizer_step

        sparsity = self
        orig = self._orig_optimizer_step

        def optimizer_step(trainer):
            trainer.scaler.unscale_(trainer.optimizer)
            sparsity.epochs = max(int(trainer.epochs), 1)
            sparsity.add_bn_l1_(de_parallel(trainer.model), int(trainer.epoch))
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=10.0)
            trainer.scaler.step(trainer.optimizer)
            trainer.scaler.update()
            trainer.optimizer.zero_grad()
            if trainer.ema:
                trainer.ema.update(trainer.model)

        BaseTrainer.optimizer_step = optimizer_step

        def restore(_trainer=None):
            BaseTrainer.optimizer_step = orig

        yolo.add_callback("on_train_end", restore)
        yolo.add_callback("teardown", restore)
