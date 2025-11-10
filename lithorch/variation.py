# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

import torch
import torch.nn as nn

import lithorch.defaults as d
from lithorch.simulation import LithographySimulator


@dataclass
class Variants:
    nominal: torch.Tensor  # [..., H, W]
    max: torch.Tensor  # [..., H, W]
    min: torch.Tensor  # [..., H, W]


@dataclass
class ProcessVariationOutput:
    aerial: Variants
    resist: Variants
    printed: Variants


class ProcessVariationSimulator(nn.Module):
    """
    Compose three LithographySimulator instances (nominal, max, min) to model
    process variations of aerial, resist, and printed outputs.
    """

    nominal_simulator: LithographySimulator
    max_simulator: LithographySimulator
    min_simulator: LithographySimulator

    def __init__(
        self,
        dose_nominal: float = d.DOSE_NOMINAL,
        dose_min: float = d.DOSE_MIN,
        dose_max: float = d.DOSE_MAX,
        resist_threshold: float = d.RESIST_THRESHOLD,
        resist_steepness: float = d.RESIST_STEEPNESS,
        print_threshold: float = d.PRINT_THRESHOLD,
        dtype: torch.dtype = d.DTYPE,
        margin: int = 0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.nominal_simulator = LithographySimulator(
            kernel_type="focus",
            dose=dose_nominal,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            print_threshold=print_threshold,
            dtype=dtype,
            margin=margin,
            device=device,
        )

        self.max_simulator = LithographySimulator(
            kernel_type="focus",
            dose=dose_max,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            print_threshold=print_threshold,
            dtype=dtype,
            margin=margin,
            device=device,
        )

        self.min_simulator = LithographySimulator(
            kernel_type="defocus",
            dose=dose_min,
            resist_threshold=resist_threshold,
            resist_steepness=resist_steepness,
            print_threshold=print_threshold,
            dtype=dtype,
            margin=margin,
            device=device,
        )

    def forward(self, mask: torch.Tensor, margin: int | None = None) -> ProcessVariationOutput:
        out_nom = self.nominal_simulator(mask=mask, margin=margin)
        out_max = self.max_simulator(mask=mask, margin=margin)
        out_min = self.min_simulator(mask=mask, margin=margin)

        aerial = Variants(nominal=out_nom.aerial, max=out_max.aerial, min=out_min.aerial)
        resist = Variants(nominal=out_nom.resist, max=out_max.resist, min=out_min.resist)
        printed = Variants(nominal=out_nom.printed, max=out_max.printed, min=out_min.printed)

        return ProcessVariationOutput(aerial=aerial, resist=resist, printed=printed)

    def get_pvb_map(self, mask: torch.Tensor, margin: int | None = None) -> torch.Tensor:
        """
        Process-variation band (PVB) map: difference between max and min printed images.
        Returns a float32 tensor shaped like the input image (and any leading batch dims).
        """
        simulation = self(mask=mask, margin=margin)
        printed_min = simulation.printed.min
        printed_max = simulation.printed.max
        return (printed_max - printed_min).to(torch.float32)

    def get_pvb_mean(self, mask: torch.Tensor, margin: int | None = None) -> torch.Tensor:
        """
        Mean PVB over spatial dimensions (keeps any leading batch dims).
        """
        pvb = self.get_pvb_map(mask=mask, margin=margin)
        return pvb.mean(dim=(-2, -1))
