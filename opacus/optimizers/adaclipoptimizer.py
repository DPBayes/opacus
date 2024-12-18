# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import torch
from torch.optim import Optimizer

from .optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _generate_noise,
    _mark_as_processed,
)


logger = logging.getLogger(__name__)


class AdaClipDPOptimizer(DPOptimizer):
    """
    :class:`~opacus.optimizers.optimizer.DPOptimizer` that implements
    adaptive clipping strategy
    https://arxiv.org/pdf/1905.03871.pdf
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        normalize_clipping: bool = False,
        optim_args: dict = None,
    ):

        assert(normalize_clipping == True), "Let us focus on the normalized version first"
        max_grad_norm = 1.0

        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            normalize_clipping=normalize_clipping,
            optim_args=optim_args,
        )

        target_unclipped_quantile = optim_args.get('target_unclipped_quantile', 0.0)
        clipbound_learning_rate = optim_args.get('clipbound_learning_rate', 1.0)
        count_threshold = optim_args.get('count_threshold', 1.0)
        max_clipbound = optim_args.get('max_clipbound', torch.inf)
        min_clipbound = optim_args.get('min_clipbound', -torch.inf)
        unclipped_num_std = optim_args.get('unclipped_num_std')
        assert (max_clipbound > min_clipbound), "max_clipbound must be larger than min_clipbound."
        self.clipbound = max_grad_norm # let we set the init value of clip bound to 1
        self.target_unclipped_quantile = target_unclipped_quantile
        self.clipbound_learning_rate = clipbound_learning_rate
        self.count_threshold = count_threshold
        self.max_clipbound = max_clipbound
        self.min_clipbound = min_clipbound
        self.unclipped_num_std = unclipped_num_std
        # Theorem 1. in  https://arxiv.org/pdf/1905.03871.pdf
        self.noise_multiplier = (
            self.noise_multiplier ** (-2) - (2 * unclipped_num_std) ** (-2)
        ) ** (-1 / 2)
        self.sample_size = 0
        self.unclipped_num = 0

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients, self.sample_size and self.unclipped_num
        """
        super().zero_grad(set_to_none)

        self.sample_size = 0
        self.unclipped_num = 0

    def clip_and_accumulate(self):
        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)

        quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        quantile_values = torch.quantile(per_sample_norms, torch.tensor(quantiles).cuda())
        for q, value in zip(quantiles, quantile_values):
            print(f"!!! quantile {int(q * 100)}%: {value.item()}")

        #print(f"max per_param_norms before clipping: {per_sample_norms.max().item()}")

        # Create a mask to determine which gradients need to be clipped based on the clipbound
        per_sample_clip_factor = torch.minimum(
            self.max_grad_norm / (per_sample_norms + 1e-6),
            torch.full_like(per_sample_norms, self.max_grad_norm / self.clipbound),
        )

        # the two lines below are the only changes
        # relative to the parent DPOptimizer class.
        self.sample_size += len(per_sample_clip_factor)
        self.unclipped_num +=  (per_sample_norms < self.clipbound * self.count_threshold).sum()

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def add_noise(self):
        super().add_noise()

        unclipped_num_noise = _generate_noise(
            std=self.unclipped_num_std,
            reference=self.unclipped_num,
            generator=self.generator,
        )

        self.unclipped_num = float(self.unclipped_num)
        self.unclipped_num += unclipped_num_noise

    def update_clipbound(self):
        """
        Update clipping bound based on unclipped fraction
        """
        unclipped_frac = self.unclipped_num / self.sample_size
        self.clipbound *= torch.exp(
            -self.clipbound_learning_rate
            * (unclipped_frac - self.target_unclipped_quantile)
        )
        if self.clipbound > self.max_clipbound:
            self.clipbound = self.max_clipbound
        elif self.clipbound < self.min_clipbound:
            self.clipbound = self.min_clipbound

        print(f"!!! unclipped_frac: {unclipped_frac}, self.target_unclipped_quantile: {self.target_unclipped_quantile}")
        print(f"!!! self.clipbound: {self.clipbound}")
        print("============================================")

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        pre_step_full = super().pre_step()
        if pre_step_full:
            self.update_clipbound()
        return pre_step_full

class LayerwiseAdaClipDPOptimizer(DPOptimizer):

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        normalize_clipping: bool = False,
        optim_args: dict = None,
    ):

        assert(normalize_clipping == True), "Let us focus on the normalized version first"
        max_grad_norm = 1.0

        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            normalize_clipping=normalize_clipping,
            optim_args=optim_args,
        )

        target_unclipped_quantile = optim_args.get('target_unclipped_quantile', 0.0)
        clipbound_learning_rate = optim_args.get('clipbound_learning_rate', 1.0)
        count_threshold = optim_args.get('count_threshold', 1.0)
        max_clipbound = optim_args.get('max_clipbound', torch.inf)
        min_clipbound = optim_args.get('min_clipbound', -torch.inf)
        unclipped_num_std = optim_args.get('unclipped_num_std')
        assert (max_clipbound > min_clipbound), "max_clipbound must be larger than min_clipbound."
        self.backbone_clipbound = max_grad_norm # Initial clip bound for backbone
        self.head_clipbound = max_grad_norm # Initial clip bound for head
        self.target_unclipped_quantile = target_unclipped_quantile
        self.clipbound_learning_rate = clipbound_learning_rate
        self.count_threshold = count_threshold
        self.max_clipbound = max_clipbound
        self.min_clipbound = min_clipbound
        self.unclipped_num_std = unclipped_num_std
        # Theorem 1. in  https://arxiv.org/pdf/1905.03871.pdf
        self.noise_multiplier = (
            self.noise_multiplier ** (-2) - (2 * unclipped_num_std) ** (-2)
        ) ** (-1 / 2)
        self.sample_size = 0
        self.unclipped_num_backbone = 0
        self.unclipped_num_head = 0

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients, self.sample_size and self.unclipped_num
        """
        super().zero_grad(set_to_none)

        self.sample_size = 0
        self.unclipped_num_backbone = 0
        self.unclipped_num_head = 0

    def ensure_base_bound(self, mean_backbone_norm, mean_head_norm):
        """
        Normalize the backbone and head norms such that their combined norm equals max_grad_norm.
        """
        factor = self.max_grad_norm / np.sqrt(mean_backbone_norm**2 + mean_head_norm**2)
        backbone_max_grad_norm = mean_backbone_norm * factor
        head_max_grad_norm = mean_head_norm * factor
        return backbone_max_grad_norm, head_max_grad_norm

    def clip_and_accumulate(self):
        per_param_norms = [
            g.view(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]
        # per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)

        # Separate backbone and head gradients
        backbone_norms = per_param_norms[:-2]
        head_norms = per_param_norms[-2:]

        per_sample_norms_backbone = torch.stack(backbone_norms, dim=1).norm(2, dim=1)
        per_sample_norms_head = torch.stack(head_norms, dim=1).norm(2, dim=1)

        mean_backbone_norm = per_sample_norms_backbone.mean().item()
        mean_head_norm = per_sample_norms_head.mean().item()

        # NOTE: now it is not private, fix it later
        print(f"  - mean_backbone_norm: {mean_backbone_norm}")
        print(f"  - mean_head_norm: {mean_head_norm}")

        backbone_max_grad_norm, head_max_grad_norm = self.ensure_base_bound(mean_backbone_norm, mean_head_norm)

        # Calculate separate clip factors based on adjusted max_grad_norms
        backbone_clip_factor = torch.minimum(
            backbone_max_grad_norm / (per_sample_norms_backbone + 1e-6),
            torch.full_like(per_sample_norms_backbone, backbone_max_grad_norm / self.backbone_clipbound),
        )

        head_clip_factor = torch.minimum(
            head_max_grad_norm / (per_sample_norms_head + 1e-6),
            torch.full_like(per_sample_norms_head, head_max_grad_norm / self.head_clipbound),
        )

        # Clip and scale gradients
        for p in self.params[:-2]:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", backbone_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

        for p in self.params[-2:]:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", head_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

        # Combine gradients into final form
        self.sample_size += len(per_sample_norms_head)
        self.unclipped_num_backbone += (
            per_sample_norms_backbone < self.backbone_clipbound * self.count_threshold
        ).sum()
        self.unclipped_num_head += (
            per_sample_norms_head < self.head_clipbound * self.count_threshold
        ).sum()

    def add_noise(self):
        super().add_noise()

        unclipped_num_noise_backbone = _generate_noise(
            std=self.unclipped_num_std,
            reference=self.unclipped_num_backbone,
            generator=self.generator,
        )

        unclipped_num_noise_head = _generate_noise(
            std=self.unclipped_num_std,
            reference=self.unclipped_num_head,
            generator=self.generator,
        )

        self.unclipped_num_backbone = float(self.unclipped_num_backbone)
        self.unclipped_num_head = float(self.unclipped_num_head)
        self.unclipped_num_backbone += unclipped_num_noise_backbone
        self.unclipped_num_head += unclipped_num_noise_head

    def update_clipbound(self):
        """
        Update clipping bound based on unclipped fraction
        """
        unclipped_frac_backbone = self.unclipped_num_backbone / self.sample_size
        unclipped_frac_head = self.unclipped_num_head / self.sample_size

        self.backbone_clipbound *= torch.exp(
            -self.clipbound_learning_rate
            * (unclipped_frac_backbone - self.target_unclipped_quantile)
        )
        self.head_clipbound *= torch.exp(
            -self.clipbound_learning_rate
            * (unclipped_frac_head - self.target_unclipped_quantile)
        )

        # Ensure bounds are within min and max limits
        self.backbone_clipbound = torch.clamp(self.backbone_clipbound, self.min_clipbound, self.max_clipbound)
        self.head_clipbound = torch.clamp(self.head_clipbound, self.min_clipbound, self.max_clipbound)

        print(f"!!!  - self.backbone_clipbound: {self.backbone_clipbound}")
        print(f"!!!  - self.head_clipbound: {self.head_clipbound}")

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        pre_step_full = super().pre_step()
        if pre_step_full:
            self.update_clipbound()
        return pre_step_full

