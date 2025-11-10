import matplotlib.pyplot as plt

import lithorch as lt


if __name__ == '__main__':

    # Load and preprocess the mask image
    image_size = 1024
    mask_url = 'https://raw.githubusercontent.com/thomashirtz/lithox/refs/heads/master/data/mask.png'
    target_url = 'https://raw.githubusercontent.com/thomashirtz/lithox/refs/heads/master/data/target.png'
    mask = lt.load_image(path_or_url=mask_url, size=image_size)
    target = lt.load_image(path_or_url=target_url, size=image_size)

    # Run the lithography simulation
    print("Running lithography simulation...")
    simulator = lt.LithographySimulator()
    simulation_result = simulator(mask)
    print("Simulation completed.")

    # Map titles to their corresponding image data
    title_to_data = {
        "Target": target,
        "Original mask": mask,
        "Aerial image": simulation_result.aerial,
        # "Resist Image": simulation_result.resist,
        "Printed image": simulation_result.printed,
    }

    # Plot the results in a 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.5), dpi=200)
    for ax, (title, data) in zip(axes.flatten(), title_to_data.items()):
        plot = ax.imshow(data, cmap="magma")
        ax.set_title(title)
        fig.colorbar(
            plot,
            ax=ax,
            orientation="vertical",
            fraction=0.046,
            pad=0.04,
        )
        ax.set_axis_off()

    plt.tight_layout()
    # plt.savefig('simulation.png', dpi=200)
    plt.show()
