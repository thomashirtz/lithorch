# Copyright (c) 2025, Thomas Hirtz
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import torch

import lithorch as lt


if __name__ == "__main__":
    # Load and preprocess the mask image
    image_size = 1024
    mask_url = "https://raw.githubusercontent.com/thomashirtz/lithox/refs/heads/master/data/mask.png"
    mask: torch.Tensor = lt.load_image(path_or_url=mask_url, size=image_size)  # (H, W), float32 in [0,1]

    # Instantiate process-variation simulator
    simulator = lt.ProcessVariationSimulator()

    # Run simulation (no grads needed)
    print("Running process-variation simulation...")
    with torch.inference_mode():
        var_out = simulator(mask)
    print("Simulation completed.")

    # Extract printed min and max (0/1 float tensors)
    prt_min: torch.Tensor = var_out.printed.min
    prt_max: torch.Tensor = var_out.printed.max

    # 3-class map: 0 = never prints, 1 = max-only, 2 = always prints
    class_map: torch.Tensor = (prt_max + prt_min).to(torch.int32)

    # Plot mask and PVB classes side by side
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.04), dpi=200)

    axes[0].imshow(mask.cpu().numpy(), cmap="gray", interpolation="nearest")
    axes[0].set_title("Mask")
    axes[0].axis("off")

    cmap = ListedColormap(["black", "red", "gray"])
    norm = BoundaryNorm([0, 1, 2, 3], cmap.N)

    im = axes[1].imshow(class_map.cpu().numpy(), cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title("Process variation")
    axes[1].axis("off")

    # Add colorbar with class labels
    cbar = fig.colorbar(im, ax=axes[1], ticks=[0.5, 1.5, 2.5], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["Never prints", "Variation band", "Always prints"])

    plt.tight_layout()
    plt.show()

    # Fraction summary
    total = class_map.numel()
    frac_never = (class_map == 0).sum().item() / total
    frac_varband = (class_map == 1).sum().item() / total
    frac_always = (class_map == 2).sum().item() / total

    print(f"Fraction never prints: {frac_never:.4f}")
    print(f"Fraction variation band: {frac_varband:.4f}")
    print(f"Fraction always prints: {frac_always:.4f}")
