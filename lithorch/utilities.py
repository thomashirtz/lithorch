# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

from importlib import resources
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image
import torch
import torch.nn.functional as F


def centered_fft_2d(data: torch.Tensor) -> torch.Tensor:
    """
    Compute a centered 2D Fast Fourier Transform over the last two dimensions.

    The zero-frequency component is shifted to the center both before and after
    the transform.

    Args:
        data (Tensor): Input of shape (..., H, W). Real or complex.

    Returns:
        Tensor: Complex tensor of the same shape as `data` containing the centered 2D FFT.
    """
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.fftn(data, dim=(-2, -1))
    data = torch.fft.fftshift(data, dim=(-2, -1))
    return data


def centered_ifft_2d(data: torch.Tensor) -> torch.Tensor:
    """
    Compute a centered 2D inverse Fast Fourier Transform over the last two dimensions.

    The zero-frequency component is shifted to the center both before and after
    the inverse transform.

    Args:
        data (Tensor): Input of shape (..., H, W). Complex or real.

    Returns:
        Tensor: Complex tensor of the same shape as `data` containing the centered 2D inverse FFT.
    """
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.ifftn(data, dim=(-2, -1))
    data = torch.fft.fftshift(data, dim=(-2, -1))
    return data


def pad_to_shape_2d(t: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """
    Zero-pad the last two (H, W) dimensions of `t` to match `shape`, keeping all leading dims.

    Padding is centered: if the difference is odd, the extra pixel goes to the
    bottom and right. Raises AssertionError if the input spatial size is larger
    than the target.

    Args:
        t (Tensor): Input of shape (..., H, W).
        shape (tuple[int, int]): Target (h_out, w_out).

    Returns:
        Tensor: Padded tensor of shape (..., h_out, w_out).
    """
    h_out, w_out = shape
    if t.ndim < 2:
        raise ValueError("Input must have at least 2 dims shaped (..., H, W).")

    *_, h_in, w_in = t.shape
    assert (
        h_in <= h_out and w_in <= w_out
    ), f"Kernel size ({h_in}×{w_in}) larger than target ({h_out}×{w_out})."

    pad_top = (h_out - h_in) // 2
    pad_bottom = h_out - h_in - pad_top
    pad_left = (w_out - w_in) // 2
    pad_right = w_out - w_in - pad_left

    # F.pad uses a flat tuple in reverse spatial order:
    # (pad_left, pad_right, pad_top, pad_bottom)
    pad = (pad_left, pad_right, pad_top, pad_bottom)
    return F.pad(t, pad, mode="constant", value=0)


def load_image(
    path_or_url: str | Path,
    size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load a grayscale image from a local path or URL, resize to (size, size), and normalize to [0, 1].

    Args:
        path_or_url (str | Path): Local filesystem path or HTTP/HTTPS URL.
        size (int): Target height and width in pixels.
        dtype (torch.dtype, optional): Output dtype (default: torch.float32).

    Returns:
        Tensor: Image tensor of shape (size, size) with values in [0, 1].
    """
    if isinstance(path_or_url, str) and path_or_url.startswith(("http://", "https://")):
        resp = requests.get(path_or_url)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
    else:
        img = Image.open(path_or_url)

    img = img.convert("L").resize((size, size), Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    return t


def crop_margin_2d(t: torch.Tensor, margin: int | tuple[int, int]) -> torch.Tensor:
    """
    Crop the last two dimensions (H, W) of a tensor by a symmetric margin.

    Args:
        t (Tensor): Input of shape (..., H, W).
        margin (int | tuple[int, int]): Single int `p` to crop `p` on all sides,
            or a tuple `(crop_h, crop_w)`.

    Returns:
        Tensor: Cropped tensor with shape (..., H - 2*crop_h, W - 2*crop_w).
    """
    if isinstance(margin, int):
        crop_h = crop_w = margin
    else:
        crop_h, crop_w = margin

    if crop_h == 0 and crop_w == 0:
        return t

    h_slice = slice(crop_h, -crop_h if crop_h != 0 else None)
    w_slice = slice(crop_w, -crop_w if crop_w != 0 else None)
    return t[..., h_slice, w_slice]


def pad_margin_2d(t: torch.Tensor, padding: int | tuple[int, int]) -> torch.Tensor:
    """
    Zero-pad the last two dimensions (H, W) of a tensor with a symmetric margin.

    Args:
        t (Tensor): Input of shape (..., H, W).
        padding (int | tuple[int, int]): Single int `p` to pad `p` on all sides,
            or a tuple `(pad_h, pad_w)`.

    Returns:
        Tensor: Padded tensor with shape (..., H + 2*pad_h, W + 2*pad_w).
    """
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    pad = (pad_w, pad_w, pad_h, pad_h)  # (left, right, top, bottom)
    return F.pad(t, pad, mode="constant", value=0)


def load_npy(
    filename: str,
    module: str | None = None,
    path: Path | None = None,
) -> torch.Tensor:
    """
    Load a .npy array via importlib.resources with an optional filesystem fallback.

    If `module` is provided, attempts to load `filename` from that package.
    If that fails (or `module` is None) and `path` is provided, loads from `path/filename`.

    Args:
        filename (str): Name of the .npy file (e.g., "focus.npy").
        module (str | None): Dotted package name to load from. If None, skip package loading.
        path (Path | None): Filesystem directory to load from as a fallback (or primary if `module` is None).

    Returns:
        Tensor: Tensor with the contents of the .npy file. Dtype matches the file's dtype.
    """
    if not module and not path:
        raise ValueError("At least one of `module` or `path` must be provided.")

    # 1) Try package resource
    if module:
        try:
            pkg_files = resources.files(module)
            resource = pkg_files / filename
            with resources.as_file(resource) as file_path:
                arr = np.load(file_path)
                return torch.from_numpy(arr.copy())
        except (ModuleNotFoundError, FileNotFoundError) as e:
            if not path:
                raise FileNotFoundError(
                    f"Could not load '{filename}' from module '{module}' and no fallback path was provided."
                ) from e

    # 2) Fallback to filesystem (or primary if no module)
    if path:
        fs_path = path / filename
        try:
            arr = np.load(fs_path)
            return torch.from_numpy(arr.copy())
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Failed to load file from filesystem path: {fs_path}") from e
