from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from baselines import gaussian_blur_3x3
from noise import add_gaussian_noise


BATCH_SIZE = 32
NUMBER_OF_IMAGES = 6
NOISE_SIGMA = 0.3

DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")


def mean_squared_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the mean squared difference between two tensors.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape"
        )

    squared_errors = (prediction - target) ** 2
    return squared_errors.mean()


def main() -> None:
    torch.manual_seed(42)

    transform = transforms.ToTensor()

    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    clean_images, labels = next(iter(test_loader))

    noisy_images, _ = add_gaussian_noise(
        clean_images,
        sigma=NOISE_SIGMA,
    )

    filtered_images = gaussian_blur_3x3(noisy_images)

    noisy_mse = mean_squared_error(
        noisy_images,
        clean_images,
    )

    filtered_mse = mean_squared_error(
        filtered_images,
        clean_images,
    )

    print("Tensor shapes")
    print("  Clean:", clean_images.shape)
    print("  Noisy:", noisy_images.shape)
    print("  Filtered:", filtered_images.shape)

    print()
    print("Reconstruction errors")
    print("  Noisy input MSE:", noisy_mse.item())
    print("  Gaussian filter MSE:", filtered_mse.item())

    if filtered_mse < noisy_mse:
        print("  The filter reduced the average squared error.")
    else:
        print("  The filter did not reduce the average squared error.") 

    make_figure(
        clean_images=clean_images,
        noisy_images=noisy_images,
        filtered_images=filtered_images,
        labels=labels,
        class_names=test_dataset.classes,
    )


def make_figure(
    clean_images: torch.Tensor,
    noisy_images: torch.Tensor,
    filtered_images: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
) -> None:
    figure, axes = plt.subplots(
        3,
        NUMBER_OF_IMAGES,
        figsize=(12, 6),
    )

    image_groups = [
        ("Clean", clean_images),
        ("Noisy", noisy_images),
        ("Gaussian filter", filtered_images),
    ]

    for row, (row_name, image_batch) in enumerate(image_groups):
        for column in range(NUMBER_OF_IMAGES):
            image = image_batch[column].squeeze(0) 
            axes[row, column].imshow(
                image,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )

            axes[row, column].axis("off")

            if row == 0:
                label = labels[column].item()
                axes[row, column].set_title(
                    class_names[label]
                )

        axes[row, 0].set_ylabel(
            row_name,
            fontsize=11,
        )

    figure.suptitle(
        f"Classical Gaussian-filter baseline, noise sigma = {NOISE_SIGMA}"
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.95) 
    )
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = FIGURE_DIR / "classical_gaussian_baseline.png"

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.show()

    print()
    print("Figure saved in:", output_path)


if __name__ == "__main__":
    main()