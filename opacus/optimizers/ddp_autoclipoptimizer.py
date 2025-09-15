# opacus/optimizers/autoclipoptimizer.py
from __future__ import annotations
import logging
from typing import Optional
import torch
from torch.optim import Optimizer

from .ddpoptimizer import DistributedDPOptimizer
from .optimizer import _check_processed_flag, _mark_as_processed

logger = logging.getLogger(__name__)


class DistributedAutoClipOptimizer(DistributedDPOptimizer):
    """
    AUTO-S (R-independent) for DDP:
      g_i -> g_i / (||g_i||_2 + γ), γ > 0
    Noise std equals `noise_multiplier` (σ from the accountant). We force max_grad_norm=1.

    As introduced by Bu et al. (2023):
        https://proceedings.neurips.cc/paper_files/paper/2023/file/8249b30d877c91611fd8c7aa6ac2b5fe-Paper-Conference.pdf

    Derived from Linzh's work on the non-DP version:
        https://github.com/DPBayes/opacus/blob/adaptive_optimizer/opacus/optimizers/autoclipoptimizer.py

    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,  # σ from accountant (sensitivity of the SUM)
        max_grad_norm: float,  # ignored; forced to 1.0
        expected_batch_size: Optional[
            int
        ],  # MUST be the **per-rank** expected Poisson size
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        normalize_clipping: bool = False,  # must remain False
        optim_args: dict | None = None,
    ):
        if normalize_clipping:
            raise AssertionError(
                "AUTO-S uses unnormalized clipping (normalize_clipping=False)."
            )

        if max_grad_norm != 1.0:
            logger.warning(
                "AutoSFixedDistributedDPOptimizer: overriding max_grad_norm=%s to 1.0.",
                max_grad_norm,
            )

        max_grad_norm = 1.0

        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,  # this is σ
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            normalize_clipping=normalize_clipping,
        )

        gamma_default = 1e-2

        self.gamma = float((optim_args or {}).get("stability_const", gamma_default))

        if self.gamma <= 0.0:
            raise ValueError("stability_const γ must be > 0.")

        logger.info(
            "AUTO-S γ=%g, world_size=%s, loss_reduction=%s",
            self.gamma,
            self.world_size,
            self.loss_reduction,
        )

    def clip_and_accumulate(self):
        """
        Apply AUTO-S: per-sample scale = 1 / (||g_i||_2 + γ), no min(1,·), no R.
        Mirrors base implementation’s empty-batch guard and uses reshape for secure_mode.
        """
        # Empty-batch guard (Poisson can produce N=0 on a rank)
        if len(self.grad_samples[0]) == 0:
            per_sample_clip_factor = torch.zeros(
                (0,), device=self.grad_samples[0].device
            )
        else:
            # Compute per-sample L2 over all params with one pass of squared sums
            total_sq = None
            for g in self.grad_samples:
                flat = g.reshape(len(g), -1)
                s = (flat * flat).sum(-1)
                total_sq = s if total_sq is None else total_sq + s

            per_sample_norms = total_sq.sqrt()
            per_sample_clip_factor = 1.0 / (per_sample_norms + self.gamma)

        # Accumulate scaled per-sample grads into summed_grad (SUM semantics)
        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)  # [N, ...]

            # be explicit in einsum to avoid silent shape mistakes
            scale = per_sample_clip_factor.to(
                dtype=grad_sample.dtype, device=grad_sample.device
            )
            grad = torch.einsum("i,i...->...", scale, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)
