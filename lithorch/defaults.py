# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

import torch


DOSE : float = 1.00
DOSE_NOMINAL: float = DOSE
DOSE_MAX: float = 1.02
DOSE_MIN: float = 0.98
RESIST_THRESHOLD: float = 0.225
RESIST_STEEPNESS: float = 50.0
PRINT_THRESHOLD: float = 0.5
DTYPE: torch.dtype = torch.float32
