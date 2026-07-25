from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from baselines import gaussian_blur_3x3
from models import (
    ConvolutionalDenoisingAutoencoder,
    FullyConnectedDenoisingAutoencoder,
)


METHOD_NAMES = [
    "Noisy input",
    "Gaussian filter",
    "FC autoencoder",
    "Conv autoencoder",
]


def load_trained_models(
    device: torch.device,
    latent_dim: int = 16,
    fc_checkpoint_path: Path = Path(
        "checkpoints/best_fc_autoencoder.pt"
    ),
    conv_checkpoint_path: Path = Path(
        "checkpoints/best_conv_autoencoder.pt"
    ),
) -> tuple[
    FullyConnectedDenoisingAutoencoder,
    ConvolutionalDenoisingAutoencoder,
]:
    """
    Construct the two autoencoders and load their trained parameters.
    """
    fc_model = FullyConnectedDenoisingAutoencoder(
        latent_dim=latent_dim,
    ).to(device)

    conv_model = ConvolutionalDenoisingAutoencoder(
        latent_dim=latent_dim,
    ).to(device)

    fc_model.load_state_dict(
        torch.load(
            fc_checkpoint_path,
            map_location=device,
        )
    )

    conv_model.load_state_dict(
        torch.load(
            conv_checkpoint_path,
            map_location=device,
        )
    )

    fc_model.eval()
    conv_model.eval()

    return fc_model, conv_model


def mean_squared_error_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Return one MSE value for every image in the batch.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape"
        )

    squared_errors = (prediction - target) ** 2

    return squared_errors.flatten(
        start_dim=1
    ).mean(dim=1)


def psnr_from_mse(
    mse_values: torch.Tensor,
) -> torch.Tensor:
    """
    Compute PSNR assuming that pixel values lie in [0, 1].
    """
    safe_mse_values = mse_values.clamp_min(1e-12)

    return 10.0 * torch.log10(
        1.0 / safe_mse_values
    )


def evaluate_corruption(
    test_loader: DataLoader,
    fc_model: FullyConnectedDenoisingAutoencoder,
    conv_model: ConvolutionalDenoisingAutoencoder,
    corruption_function: Callable[
        [torch.Tensor],
        torch.Tensor,
    ],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Evaluate all methods using one corruption function.
    """
    accumulated_metrics = {
        method_name: {
            "mse_sum": 0.0,
            "psnr_sum": 0.0,
        }
        for method_name in METHOD_NAMES
    }

    total_images = 0

    with torch.no_grad():
        for clean_images, _ in test_loader:
            clean_images = clean_images.to(device)

            noisy_images = corruption_function(
                clean_images
            )

            predictions = {
                "Noisy input": noisy_images,
                "Gaussian filter": gaussian_blur_3x3(
                    noisy_images
                ),
                "FC autoencoder": fc_model(
                    noisy_images
                ),
                "Conv autoencoder": conv_model(
                    noisy_images
                ),
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

            total_images += clean_images.shape[0]

    return {
        method_name: {
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
        for method_name in METHOD_NAMES
    }