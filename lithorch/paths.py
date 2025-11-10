# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path


CURRENT_FILE_PATH = Path(__file__).resolve()
PACKAGE_DIRECTORY = CURRENT_FILE_PATH.parent
KERNELS_DIRECTORY = PACKAGE_DIRECTORY / "kernels"
SCALES_DIRECTORY = PACKAGE_DIRECTORY / "scales"
