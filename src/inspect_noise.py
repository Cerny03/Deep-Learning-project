from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from noise import add_gaussian_noise


BATCH_SIZE = 32
NUMBER_OF_IMAGES = 6
DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")

NOISE_LEVELS = [0.1, 0.3, 0.5]

def main() -> None:
    torch.manual_seed(42)

    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    print("Clean batch")
    print("  Shape:", images.shape)
    print("  Minimum:", images.min().item())
    print("  Maximum:", images.max().item())

    corrupted_batches = []

    for sigma in NOISE_LEVELS:
        noisy_images, noise = add_gaussian_noise(images, sigma=sigma)

        corrupted_batches.append(noisy_images)

        print()
        print(f"Gaussian noise with sigma = {sigma}")
        print("  Noise shape:", noise.shape)
        print("  Empirical noise mean:", noise.mean().item())
        print("  Empirical noise standard deviation:", noise.std().item())
        print("  Raw noise minimum:", noise.min().item())
        print("  Raw noise maximum:", noise.max().item())
        print("  Noisy image minimum:", noisy_images.min().item())
        print("  Noisy image maximum:", noisy_images.max().item())

        assert noisy_images.shape == images.shape 
        assert noise.shape == images.shape
        assert noisy_images.min().item() >= 0.0
        assert noisy_images.max().item() <= 1.0

    make_figure(
        clean_images=images,
        corrupted_batches=corrupted_batches,
        labels=labels,
        class_names=train_dataset.classes,
    )

def make_figure(
    clean_images: torch.Tensor,
    corrupted_batches: list[torch.Tensor],
    labels: torch.Tensor,
    class_names: list[str],
) -> None:
    number_of_rows = 1 + len(corrupted_batches)

    figure, axes = plt.subplots(
        number_of_rows,
        NUMBER_OF_IMAGES,
        figsize=(12, 7),
    )

    for column in range(NUMBER_OF_IMAGES):
        clean_image = clean_images[column].squeeze(0)
        label = labels[column].item()

        axes[0, column].imshow(clean_image, cmap="gray")
        axes[0, column].set_title(class_names[label])
        axes[0, column].axis("off")

    axes[0, 0].set_ylabel("Clean", fontsize=11)

    for row, (sigma, noisy_images) in enumerate(
        zip(NOISE_LEVELS, corrupted_batches),
        start=1, 
    ):
        for column in range(NUMBER_OF_IMAGES):
            noisy_image = noisy_images[column].squeeze(0)

            axes[row, column].imshow(
                noisy_image,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, column].axis("off")

        axes[row, 0].set_ylabel(
            f"sigma = {sigma}",
            fontsize=11,
        )

    figure.suptitle("Effect of Gaussian noise on Fashion-MNIST images")
    figure.tight_layout()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURE_DIR / "gaussian_noise_levels.png"
    figure.savefig(output_path, dpi=150)
    plt.show()

    print()
    print("Figure saved in:", output_path)


if __name__ == "__main__":
    main()