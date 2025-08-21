from __future__ import annotations
import logging
from typing import Callable, Optional

import torch
from torch.optim import Optimizer

from .optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _mark_as_processed,
)

logger = logging.getLogger(__name__)


class AutoSFixedDPOptimizer(DPOptimizer):
    """
    R-independent AUTO-S clipping (arXiv:2206.07136 §4):
        g_i -> g_i / (||g_i||_2 + γ), with γ > 0.
    Noise std equals `noise_multiplier` (σ from accountant). We force max_grad_norm=1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,  # set σ from accountant directly
        max_grad_norm: float,  # ignored; forced to 1.0 to keep R-independent
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        normalize_clipping: bool = False,  # must be False
        optim_args: dict | None = None,
    ):
        if normalize_clipping:
            raise AssertionError(
                "AUTO-S uses unnormalized clipping (normalize_clipping=False)."
            )

        # Force R = 1 to make noise std = σ and remove any dependence on R.
        if max_grad_norm != 1.0:
            logger.warning(
                "AutoSFixedDPOptimizer: overriding max_grad_norm=%s to 1.0 for R-independence.",
                max_grad_norm,
            )
        max_grad_norm = 1.0

        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,  # this is σ
            max_grad_norm=max_grad_norm,  # fixed to 1.0
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            normalize_clipping=normalize_clipping,
            optim_args=optim_args,
        )

        gamma_default = 1e-2  # §4: γ=0.01 as default
        self.gamma = float((optim_args or {}).get("stability_const", gamma_default))

        if self.gamma <= 0.0:
            raise ValueError("stability_const γ must be > 0.")

    def clip_and_accumulate(self):
        """
        Apply R-independent AUTO-S: clip factor = 1 / (||g_i||_2 + γ).
        """
        # per-sample L2 norms over all params
        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)

        # No min(1, ·), no R; pure 1 / (||g_i|| + γ)
        per_sample_clip_factor = 1.0 / (per_sample_norms + self.gamma)

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)  # shape: [N, ...]
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self):
        """
        Keep DPOptimizer's noise addition. With max_grad_norm=1.0, the noise std is exactly σ.
        """
        super().add_noise()

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        return super().pre_step(closure)
