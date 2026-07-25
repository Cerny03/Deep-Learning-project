import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from baselines import gaussian_blur_3x3
from models import (
    ConvolutionalDenoisingAutoencoder,
    FullyConnectedDenoisingAutoencoder,
)
from noise import add_gaussian_noise


BATCH_SIZE = 32
LATENT_DIM = 16
NOISE_SIGMA = 0.3
NUMBER_OF_IMAGES = 6

DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")
RESULTS_DIR = Path("results")

FC_CHECKPOINT_PATH = Path(
    "checkpoints/best_fc_autoencoder.pt"
)

CONV_CHECKPOINT_PATH = Path(
    "checkpoints/best_conv_autoencoder.pt"
)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the trainable parameters of a model.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def mean_squared_error_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute one MSE value for each image in the batch.
    Input shape:
        [B, C, H, W]
    Output shape:
        [B]
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape"
        )

    squared_errors = (prediction - target) ** 2

    squared_errors = squared_errors.flatten(
        start_dim=1
    )
    return squared_errors.mean(dim=1)

def psnr_from_mse(
    mse_values: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the PSNR of each image.
    Pixel values are assumed to lie in [0, 1].
    """
    safe_mse_values = mse_values.clamp_min(1e-12)
    return 10.0 * torch.log10(
        1.0 / safe_mse_values
    )


def save_results_csv(
    results: dict[str, dict[str, float]],
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "denoising_comparison_sigma_0.3.csv"
    )

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "method",
                "mse",
                "psnr_db",
            ],
        )

        writer.writeheader()

        for method_name, metrics in results.items():
            writer.writerow(
                {
                    "method": method_name,
                    "mse": metrics["mse"],
                    "psnr_db": metrics["psnr"],
                }
            )

    print("Results saved in:", output_path)


def save_comparison_figure(
    clean_images: torch.Tensor,
    noisy_images: torch.Tensor,
    filtered_images: torch.Tensor,
    fc_reconstructions: torch.Tensor,
    conv_reconstructions: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
) -> None:
    """
    Save a qualitative comparison of all methods.
    """
    image_groups = [
        ("Clean", clean_images),
        ("Noisy", noisy_images),
        ("Gaussian filter", filtered_images),
        ("FC autoencoder", fc_reconstructions),
        ("Conv autoencoder", conv_reconstructions),
    ]

    number_of_rows = len(image_groups)

    figure, axes = plt.subplots(
        number_of_rows,
        NUMBER_OF_IMAGES,
        figsize=(12, 9),
    )

    for row, (row_name, image_batch) in enumerate(
        image_groups
    ):
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
            fontsize=10,
        )

    figure.suptitle(
        "Denoising comparison on Fashion-MNIST, "
        f"noise sigma = {NOISE_SIGMA}"
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.96)
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURE_DIR
        / "all_models_comparison_sigma_0.3.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.show()

    print("Comparison figure saved in:", output_path)


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

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

    fc_model = FullyConnectedDenoisingAutoencoder(
        latent_dim=LATENT_DIM,
    ).to(device)

    conv_model = ConvolutionalDenoisingAutoencoder(
        latent_dim=LATENT_DIM,
    ).to(device)

    fc_state_dict = torch.load(
        FC_CHECKPOINT_PATH,
        map_location=device,
    )

    conv_state_dict = torch.load(
        CONV_CHECKPOINT_PATH,
        map_location=device,
    )

    fc_model.load_state_dict(fc_state_dict)
    conv_model.load_state_dict(conv_state_dict)

    fc_model.eval()
    conv_model.eval()

    print()
    print("Trainable parameters")
    print(
        "  FC autoencoder:",
        count_parameters(fc_model),
    )
    print(
        "  Conv autoencoder:",
        count_parameters(conv_model),
    )

    method_names = [
        "Noisy input",
        "Gaussian filter",
        "FC autoencoder",
        "Conv autoencoder",
    ]

    accumulated_metrics = {
        method_name: {
            "mse_sum": 0.0,
            "psnr_sum": 0.0,
        }
        for method_name in method_names
    }

    total_images = 0
    example_data = None

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    with torch.no_grad():
        for clean_images, labels in test_loader:
            clean_images = clean_images.to(device)

            noisy_images, _ = add_gaussian_noise(
                clean_images,
                sigma=NOISE_SIGMA,
            )

            filtered_images = gaussian_blur_3x3(
                noisy_images
            )

            fc_reconstructions = fc_model(
                noisy_images
            )

            conv_reconstructions = conv_model(
                noisy_images
            )

            predictions = {
                "Noisy input": noisy_images,
                "Gaussian filter": filtered_images,
                "FC autoencoder": fc_reconstructions,
                "Conv autoencoder": conv_reconstructions,
            }

            for method_name, prediction in predictions.items():
                mse_values = mean_squared_error_per_image(
                    prediction,
                    clean_images,
                )

                psnr_values = psnr_from_mse(
                    mse_values
                )

                accumulated_metrics[
                    method_name
                ]["mse_sum"] += mse_values.sum().item()

                accumulated_metrics[
                    method_name
                ]["psnr_sum"] += psnr_values.sum().item()

            current_batch_size = clean_images.shape[0]
            total_images += current_batch_size

            if example_data is None:
                example_data = (
                    clean_images.cpu(),
                    noisy_images.cpu(),
                    filtered_images.cpu(),
                    fc_reconstructions.cpu(),
                    conv_reconstructions.cpu(),
                    labels,
                )

    results = {}

    for method_name in method_names:
        results[method_name] = {
            "mse": (
                accumulated_metrics[
                    method_name
                ]["mse_sum"]
                / total_images
            ),
            "psnr": (
                accumulated_metrics[
                    method_name
                ]["psnr_sum"]
                / total_images
            ),
        }

    print()
    print(
        f"Test results with noise sigma = {NOISE_SIGMA}"
    )

    print(
        f"{'Method':<22}"
        f"{'MSE':>12}"
        f"{'PSNR (dB)':>14}"
    )

    print("-" * 48)

    for method_name, metrics in results.items():
        print(
            f"{method_name:<22}"
            f"{metrics['mse']:>12.6f}"
            f"{metrics['psnr']:>14.3f}"
        )

    fc_mse = results["FC autoencoder"]["mse"]
    conv_mse = results["Conv autoencoder"]["mse"]

    relative_difference = (
        1.0 - conv_mse / fc_mse
    ) * 100.0

    print()
    print(
        "Conv autoencoder MSE improvement over "
        f"FC autoencoder: {relative_difference:.2f}%"
    )

    save_results_csv(results)

    if example_data is None:
        raise RuntimeError("The test loader was empty")

    (
        clean_examples,
        noisy_examples,
        filtered_examples,
        fc_examples,
        conv_examples,
        example_labels,
    ) = example_data

    save_comparison_figure(
        clean_images=clean_examples,
        noisy_images=noisy_examples,
        filtered_images=filtered_examples,
        fc_reconstructions=fc_examples,
        conv_reconstructions=conv_examples,
        labels=example_labels,
        class_names=test_dataset.classes,
    )


if __name__ == "__main__":
    main()