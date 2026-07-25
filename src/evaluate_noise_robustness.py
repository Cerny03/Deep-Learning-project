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

TRAIN_NOISE_SIGMA = 0.3

TEST_NOISE_LEVELS = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
]

EVALUATION_SEED = 12345

DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")
RESULTS_DIR = Path("results")

FC_CHECKPOINT_PATH = Path(
    "checkpoints/best_fc_autoencoder.pt"
)

CONV_CHECKPOINT_PATH = Path(
    "checkpoints/best_conv_autoencoder.pt"
)

METHOD_NAMES = [
    "Noisy input",
    "Gaussian filter",
    "FC autoencoder",
    "Conv autoencoder",
]


def mean_squared_error_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute one MSE value for every image in a batch.

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
    Compute one PSNR value for every image.

    Pixel values are assumed to lie in [0, 1].
    """
    safe_mse_values = mse_values.clamp_min(1e-12)

    return 10.0 * torch.log10(
        1.0 / safe_mse_values
    )


def load_models(
    device: torch.device,
) -> tuple[
    FullyConnectedDenoisingAutoencoder,
    ConvolutionalDenoisingAutoencoder,
]:
    """
    Construct the two models and load their best checkpoints.
    """
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

    return fc_model, conv_model


def evaluate_at_noise_level(
    test_loader: DataLoader,
    fc_model: FullyConnectedDenoisingAutoencoder,
    conv_model: ConvolutionalDenoisingAutoencoder,
    sigma: float,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Evaluate all methods at one noise intensity.
    """
    accumulated_metrics = {
        method_name: {
            "mse_sum": 0.0,
            "psnr_sum": 0.0,
        }
        for method_name in METHOD_NAMES
    }

    total_images = 0

    torch.manual_seed(EVALUATION_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EVALUATION_SEED)

    with torch.no_grad():
        for clean_images, _ in test_loader:
            clean_images = clean_images.to(device)

            noisy_images, _ = add_gaussian_noise(
                clean_images,
                sigma=sigma,
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

    results = {}

    for method_name in METHOD_NAMES:
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

    return results


def save_results_csv(
    results_by_sigma: dict[
        float,
        dict[str, dict[str, float]],
    ],
) -> None:
    """
    Save one row for every noise level and method.
    """
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "noise_robustness.csv"
    )

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "test_sigma",
                "method",
                "mse",
                "psnr_db",
            ],
        )

        writer.writeheader()

        for sigma in TEST_NOISE_LEVELS:
            for method_name in METHOD_NAMES:
                metrics = results_by_sigma[
                    sigma
                ][method_name]

                writer.writerow(
                    {
                        "test_sigma": sigma,
                        "method": method_name,
                        "mse": metrics["mse"],
                        "psnr_db": metrics["psnr"],
                    }
                )

    print("Results saved in:", output_path)


def save_metric_plot(
    results_by_sigma: dict[
        float,
        dict[str, dict[str, float]],
    ],
    metric_name: str,
    y_label: str,
    title: str,
    filename: str,
) -> None:
    """
    Plot one metric as a function of the test noise level.
    """
    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    for method_name in METHOD_NAMES:
        metric_values = [
            results_by_sigma[
                sigma
            ][method_name][metric_name]
            for sigma in TEST_NOISE_LEVELS
        ]

        axis.plot(
            TEST_NOISE_LEVELS,
            metric_values,
            marker="o",
            label=method_name,
        )

    axis.axvline(
        TRAIN_NOISE_SIGMA,
        linestyle="--",
        label="Training noise level",
    )

    axis.set_xlabel(
        "Test Gaussian noise standard deviation"
    )

    axis.set_ylabel(y_label)
    axis.set_title(title)

    axis.set_xticks(TEST_NOISE_LEVELS)
    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = FIGURE_DIR / filename

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.show()

    print("Figure saved in:", output_path)


def print_results_table(
    sigma: float,
    results: dict[str, dict[str, float]],
) -> None:
    """
    Print the results obtained at one noise intensity.
    """
    print()
    print(f"Test noise sigma = {sigma}")

    print(
        f"{'Method':<22}"
        f"{'MSE':>12}"
        f"{'PSNR (dB)':>14}"
    )

    print("-" * 48)

    for method_name in METHOD_NAMES:
        metrics = results[method_name]

        print(
            f"{method_name:<22}"
            f"{metrics['mse']:>12.6f}"
            f"{metrics['psnr']:>14.3f}"
        )

    fc_mse = results[
        "FC autoencoder"
    ]["mse"]

    conv_mse = results[
        "Conv autoencoder"
    ]["mse"]

    conv_improvement = (
        1.0 - conv_mse / fc_mse
    ) * 100.0

    print(
        "Conv autoencoder improvement over "
        f"FC autoencoder: {conv_improvement:.2f}%"
    )


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

    fc_model, conv_model = load_models(
        device=device
    )

    results_by_sigma = {}

    for sigma in TEST_NOISE_LEVELS:
        results = evaluate_at_noise_level(
            test_loader=test_loader,
            fc_model=fc_model,
            conv_model=conv_model,
            sigma=sigma,
            device=device,
        )

        results_by_sigma[sigma] = results

        print_results_table(
            sigma=sigma,
            results=results,
        )

    save_results_csv(results_by_sigma)

    save_metric_plot(
        results_by_sigma=results_by_sigma,
        metric_name="mse",
        y_label="Mean squared error",
        title=(
            "Robustness to Gaussian noise intensity"
        ),
        filename="noise_robustness_mse.png",
    )

    save_metric_plot(
        results_by_sigma=results_by_sigma,
        metric_name="psnr",
        y_label="PSNR (dB)",
        title=(
            "PSNR under different Gaussian noise levels"
        ),
        filename="noise_robustness_psnr.png",
    )


if __name__ == "__main__":
    main()