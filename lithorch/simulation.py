# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn

import lithorch.defaults as d
import lithorch.paths as p
from lithorch.utilities import centered_fft_2d, centered_ifft_2d, crop_margin_2d, load_npy, pad_margin_2d, pad_to_shape_2d


def _convolve_frequency_domain(
    image_stack: torch.Tensor,          # [..., K, H, W] or [..., 1, H, W], real or complex
    kernels_fourier: torch.Tensor,      # [K, Hk, Wk], complex
) -> torch.Tensor:
    image_stack_c = image_stack.to(torch.complex64)
    H, W = image_stack_c.shape[-2:]
    kernels_padded = pad_to_shape_2d(kernels_fourier, (H, W))  # [K, H, W], complex

    stack_ft = centered_fft_2d(image_stack_c)                  # [..., K, H, W]
    bshape = (1,) * (stack_ft.ndim - 3) + kernels_padded.shape
    product_ft = stack_ft * kernels_padded.reshape(bshape)
    return centered_ifft_2d(product_ft)                        # [..., K, H, W], complex


class _SimulateAerialFromMask(torch.autograd.Function):
    """
    I = sum_k scales[k] * | F^{-1}( F(dose * mask) * kernels_fourier[k] ) |^2
    Grad flows only to `mask` (others are treated as constants).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        mask: torch.Tensor,                 # [..., H, W], real
        dose: float,                        # scalar Python float
        kernels_fourier: torch.Tensor,      # [K, Hk, Wk], complex
        kernels_fourier_ct: torch.Tensor,   # [K, Hk, Wk], complex
        scales: torch.Tensor,               # [K], real >= 0
    ) -> torch.Tensor:                      # [..., H, W], real
        # stop-gradient for constants (match JAX)
        k = kernels_fourier.detach()
        k_ct = kernels_fourier_ct.detach()
        s = scales.detach()

        dose_f = float(dose)
        dosed_mask = (dose_f * mask).to(torch.float32)

        fields = _convolve_frequency_domain(dosed_mask.unsqueeze(-3), k)  # [..., K, H, W]
        intensities = torch.abs(fields) ** 2                               # real
        y = torch.sum(s[..., None, None] * intensities, dim=-3)            # [..., H, W]

        # save tensors; keep dose as float on ctx
        ctx.save_for_backward(dosed_mask, fields, k, k_ct, s)
        ctx.dose = dose_f
        return y

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_aerial: torch.Tensor, # [..., H, W], real
    ) -> Tuple[torch.Tensor | None, ...]:
        dosed_mask, fields_main, k, k_ct, s = ctx.saved_tensors
        grad = grad_aerial.unsqueeze(-3)      # [..., 1, H, W], real

        fields_ct = _convolve_frequency_domain(dosed_mask.unsqueeze(-3), k_ct)
        term1 = _convolve_frequency_domain(fields_ct * grad, k)
        term2 = _convolve_frequency_domain(fields_main * grad, k_ct)

        summed = torch.sum(s[..., None, None] * (term1 + term2), dim=-3)   # complex
        grad_mask = (ctx.dose * summed.real).to(dosed_mask.dtype)

        # grads for: mask, dose, kernels_fourier, kernels_fourier_ct, scales
        return grad_mask, None, None, None, None


@dataclass
class SimulationOutput:
    aerial: torch.Tensor     # [..., H, W], real
    resist: torch.Tensor     # [..., H, W], real in [0,1]
    printed: torch.Tensor    # [..., H, W], 0/1


class LithographySimulator(nn.Module):
    def __init__(
        self,
        kernel_type: Literal["focus", "defocus"] = "focus",
        *,
        dose: float = d.DOSE,
        resist_threshold: float = d.RESIST_THRESHOLD,
        resist_steepness: float = d.RESIST_STEEPNESS,
        print_threshold: float = d.PRINT_THRESHOLD,
        dtype: torch.dtype = d.DTYPE,
        trainable: bool = False,   # kept for API parity; grads still only to mask
        margin: int = 0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.kernel_type = kernel_type
        self.dose = float(dose)
        self.margin = int(margin)
        self.resist_threshold = float(resist_threshold)
        self.resist_steepness = float(resist_steepness)
        self.print_threshold = float(print_threshold)
        self.dtype = dtype
        self.trainable = bool(trainable)

        k = load_npy(f"{kernel_type}.npy", module="lithox.kernels", path=p.KERNELS_DIRECTORY)
        k_ct = load_npy(f"{kernel_type}_ct.npy", module="lithox.kernels", path=p.KERNELS_DIRECTORY)
        s = load_npy(f"{kernel_type}.npy", module="lithox.scales", path=p.SCALES_DIRECTORY)

        self.register_buffer("kernels",    k.to(device=device, dtype=torch.complex64), persistent=True)
        self.register_buffer("kernels_ct", k_ct.to(device=device, dtype=torch.complex64), persistent=True)
        self.register_buffer("scales",     s.to(device=device, dtype=torch.float32),     persistent=True)

    def forward(self, mask: torch.Tensor, margin: int | None = None) -> SimulationOutput:
        m = self.margin if margin is None else int(margin)
        x = pad_margin_2d(mask, m) if m > 0 else mask

        aerial = self.simulate_aerial_from_mask(x, margin=0)
        resist = self.simulate_resist_from_aerial(aerial)
        printed = self.simulate_printed_from_resist(resist)

        if m > 0:
            aerial = crop_margin_2d(aerial, m)
            resist = crop_margin_2d(resist, m)
            printed = crop_margin_2d(printed, m)

        return SimulationOutput(aerial=aerial, resist=resist, printed=printed)

    def simulate_aerial_from_mask(self, mask: torch.Tensor, margin: int | None = None) -> torch.Tensor:
        m = self.margin if margin is None else int(margin)
        x = pad_margin_2d(mask, m) if m > 0 else mask

        y = _SimulateAerialFromMask.apply(
            x.to(self.dtype),           # mask
            self.dose,                  # dose as float
            self.kernels.detach(),      # stop-grad for constants
            self.kernels_ct.detach(),
            self.scales.detach(),
        )
        return crop_margin_2d(y, m) if m > 0 else y

    def simulate_resist_from_aerial(self, aerial: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.resist_steepness * (aerial - self.resist_threshold)).to(aerial.dtype)

    def simulate_printed_from_resist(self, resist: torch.Tensor) -> torch.Tensor:
        return (resist > self.print_threshold).to(resist.dtype)

    @classmethod
    def nominal(cls, **overrides) -> "LithographySimulator":
        return cls(kernel_type="focus", dose=d.DOSE_NOMINAL, **overrides)

    @classmethod
    def maximum(cls, **overrides) -> "LithographySimulator":
        return cls(kernel_type="focus", dose=d.DOSE_MAX, **overrides)

    @classmethod
    def minimum(cls, **overrides) -> "LithographySimulator":
        return cls(kernel_type="defocus", dose=d.DOSE_MIN, **overrides)
