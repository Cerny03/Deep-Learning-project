from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import (
    ConvolutionalDenoisingAutoencoder,
    FullyConnectedDenoisingAutoencoder,
)
from noise import add_gaussian_noise
from representation_utils import (
    collect_latent_representations,
    compute_latent_recovery_metrics,
    save_latent_pca_figure,
    shared_pca_projection,
)


BATCH_SIZE = 32
LATENT_DIM = 16
NOISE_SIGMA = 0.3

ANALYSIS_SEED = 12345
NUMBER_OF_PCA_POINTS = 2000

DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")

FC_CHECKPOINT_PATH = Path(
    "checkpoints/best_fc_autoencoder.pt"
)

CONV_CHECKPOINT_PATH = Path(
    "checkpoints/best_conv_autoencoder.pt"
)


def gaussian_corruption(
    images: torch.Tensor,
) -> torch.Tensor:
    """
    Add Gaussian noise with the same intensity used
    during autoencoder training.
    """
    noisy_images, _ = add_gaussian_noise(
        images,
        sigma=NOISE_SIGMA,
    )

    return noisy_images


def set_random_seed() -> None:
    """
    Reset the PyTorch random generator.
    """
    torch.manual_seed(
        ANALYSIS_SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            ANALYSIS_SEED
        )


def analyze_model(
    model_name: str,
    output_filename: str,
    model: torch.nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> None:
    """
    Extract, measure, and visualize the latent representations
    produced by one autoencoder.
    """
    set_random_seed()

    representations = (
        collect_latent_representations(
            model=model,
            data_loader=test_loader,
            corruption_function=(
                gaussian_corruption
            ),
            device=device,
        )
    )

    metrics = compute_latent_recovery_metrics(
        representations
    )

    print()
    print(model_name)

    print(
        "  Mean distance clean-noisy:",
        f"{metrics['clean_noisy_distance']:.6f}",
    )

    print(
        "  Mean distance clean-reconstructed:",
        (
            f"{metrics['clean_reconstructed_distance']:.6f}"
        ),
    )

    print(
        "  Relative latent recovery:",
        (
            f"{100.0 * metrics['relative_recovery']:.2f}%"
        ),
    )

    print(
        "  Images with improved latent distance:",
        (
            f"{100.0 * metrics['improved_fraction']:.2f}%"
        ),
    )

    number_of_available_images = (
        representations["labels"].shape[0]
    )

    number_of_selected_images = min(
        NUMBER_OF_PCA_POINTS,
        number_of_available_images,
    )

    index_generator = (
        torch.Generator().manual_seed(
            ANALYSIS_SEED
        )
    )

    random_indices = torch.randperm(
        number_of_available_images,
        generator=index_generator,
    )

    selected_indices = random_indices[
        :number_of_selected_images
    ]

    selected_clean_representations = (
        representations["clean"][
            selected_indices
        ]
    )

    selected_noisy_representations = (
        representations["noisy"][
            selected_indices
        ]
    )

    selected_reconstructed_representations = (
        representations["reconstructed"][
            selected_indices
        ]
    )

    selected_labels = (
        representations["labels"][
            selected_indices
        ]
    )

    set_random_seed()

    (
        clean_projection,
        noisy_projection,
        reconstructed_projection,
    ) = shared_pca_projection(
        clean_representations=(
            selected_clean_representations
        ),
        noisy_representations=(
            selected_noisy_representations
        ),
        reconstructed_representations=(
            selected_reconstructed_representations
        ),
    )

    output_path = (
        FIGURE_DIR
        / output_filename
    )

    save_latent_pca_figure(
        clean_projection=clean_projection,
        noisy_projection=noisy_projection,
        reconstructed_projection=(
            reconstructed_projection
        ),
        labels=selected_labels,
        class_names=class_names,
        title=(
            f"{model_name}: latent representations "
            f"with Gaussian noise sigma = {NOISE_SIGMA}"
        ),
        output_path=output_path,
    )


def main() -> None:
    set_random_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

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

    fully_connected_model = (
        FullyConnectedDenoisingAutoencoder(
            latent_dim=LATENT_DIM,
        )
    )

    fully_connected_model = (
        fully_connected_model.to(
            device
        )
    )

    convolutional_model = (
        ConvolutionalDenoisingAutoencoder(
            latent_dim=LATENT_DIM,
        )
    )

    convolutional_model = (
        convolutional_model.to(
            device
        )
    )

    fully_connected_state_dict = torch.load(
        FC_CHECKPOINT_PATH,
        map_location=device,
    )

    convolutional_state_dict = torch.load(
        CONV_CHECKPOINT_PATH,
        map_location=device,
    )

    fully_connected_model.load_state_dict(
        fully_connected_state_dict
    )

    convolutional_model.load_state_dict(
        convolutional_state_dict
    )

    fully_connected_model.eval()
    convolutional_model.eval()

    analyze_model(
        model_name=(
            "Fully connected autoencoder"
        ),
        output_filename=(
            "fc_latent_pca.png"
        ),
        model=fully_connected_model,
        test_loader=test_loader,
        class_names=test_dataset.classes,
        device=device,
    )

    analyze_model(
        model_name=(
            "Convolutional autoencoder"
        ),
        output_filename=(
            "conv_latent_pca.png"
        ),
        model=convolutional_model,
        test_loader=test_loader,
        class_names=test_dataset.classes,
        device=device,
    )


if __name__ == "__main__":
    main()