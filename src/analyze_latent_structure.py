import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import PercentFormatter
from torch.utils.data import (
    DataLoader,
    Subset,
)
from torchvision import datasets, transforms

from models import (
    ConvolutionalDenoisingAutoencoder,
    FullyConnectedDenoisingAutoencoder,
)
from noise import add_gaussian_noise
from representation_utils import (
    collect_clean_latent_representations,
    collect_latent_representations,
    knn_classification_accuracy,
    nearest_neighbor_indices,
    neighborhood_overlap_per_sample,
    standardize_using_reference,
)


BATCH_SIZE = 32
LATENT_DIM = 16
NOISE_SIGMA = 0.3

ANALYSIS_SEED = 12345

NUMBER_OF_REFERENCE_IMAGES = 5000
NUMBER_OF_QUERY_IMAGES = 2000

NUMBER_OF_CLASSIFICATION_NEIGHBORS = 5
NUMBER_OF_STRUCTURE_NEIGHBORS = 10

DISTANCE_BATCH_SIZE = 256

DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")
RESULTS_DIR = Path("results")

FC_CHECKPOINT_PATH = Path(
    "checkpoints/best_fc_autoencoder.pt"
)

CONV_CHECKPOINT_PATH = Path(
    "checkpoints/best_conv_autoencoder.pt"
)


def set_random_seed() -> None:
    torch.manual_seed(
        ANALYSIS_SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            ANALYSIS_SEED
        )


def gaussian_corruption(
    images: torch.Tensor,
) -> torch.Tensor:
    noisy_images, _ = add_gaussian_noise(
        images,
        sigma=NOISE_SIGMA,
    )

    return noisy_images


def create_random_subset(
    dataset,
    number_of_samples: int,
    seed: int,
) -> Subset:
    """
    Select a reproducible random subset of a dataset.
    """
    if number_of_samples > len(dataset):
        raise ValueError(
            "number_of_samples cannot exceed the dataset size"
        )

    generator = torch.Generator().manual_seed(
        seed
    )

    selected_indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:number_of_samples]

    return Subset(
        dataset,
        selected_indices.tolist(),
    )


def analyze_model(
    model_name: str,
    model: torch.nn.Module,
    reference_loader: DataLoader,
    query_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Analyze semantic classification and local-neighborhood
    preservation for one autoencoder.
    """
    model.eval()

    reference_latent, reference_labels = (
        collect_clean_latent_representations(
            model=model,
            data_loader=reference_loader,
            device=device,
        )
    )

    set_random_seed()

    query_representations = (
        collect_latent_representations(
            model=model,
            data_loader=query_loader,
            corruption_function=(
                gaussian_corruption
            ),
            device=device,
        )
    )

    query_clean = query_representations[
        "clean"
    ]

    query_noisy = query_representations[
        "noisy"
    ]

    query_reconstructed = (
        query_representations[
            "reconstructed"
        ]
    )

    query_labels = query_representations[
        "labels"
    ]

    (
        standardized_reference,
        standardized_clean,
        standardized_noisy,
        standardized_reconstructed,
    ) = standardize_using_reference(
        reference_latent,
        query_clean,
        query_noisy,
        query_reconstructed,
    )

    clean_accuracy = (
        knn_classification_accuracy(
            reference_representations=(
                standardized_reference
            ),
            reference_labels=reference_labels,
            query_representations=(
                standardized_clean
            ),
            query_labels=query_labels,
            number_of_neighbors=(
                NUMBER_OF_CLASSIFICATION_NEIGHBORS
            ),
            distance_batch_size=(
                DISTANCE_BATCH_SIZE
            ),
        )
    )

    noisy_accuracy = (
        knn_classification_accuracy(
            reference_representations=(
                standardized_reference
            ),
            reference_labels=reference_labels,
            query_representations=(
                standardized_noisy
            ),
            query_labels=query_labels,
            number_of_neighbors=(
                NUMBER_OF_CLASSIFICATION_NEIGHBORS
            ),
            distance_batch_size=(
                DISTANCE_BATCH_SIZE
            ),
        )
    )

    reconstructed_accuracy = (
        knn_classification_accuracy(
            reference_representations=(
                standardized_reference
            ),
            reference_labels=reference_labels,
            query_representations=(
                standardized_reconstructed
            ),
            query_labels=query_labels,
            number_of_neighbors=(
                NUMBER_OF_CLASSIFICATION_NEIGHBORS
            ),
            distance_batch_size=(
                DISTANCE_BATCH_SIZE
            ),
        )
    )

    clean_neighbor_indices = (
        nearest_neighbor_indices(
            representations=standardized_clean,
            number_of_neighbors=(
                NUMBER_OF_STRUCTURE_NEIGHBORS
            ),
            distance_batch_size=(
                DISTANCE_BATCH_SIZE
            ),
        )
    )

    noisy_neighbor_indices = (
        nearest_neighbor_indices(
            representations=standardized_noisy,
            number_of_neighbors=(
                NUMBER_OF_STRUCTURE_NEIGHBORS
            ),
            distance_batch_size=(
                DISTANCE_BATCH_SIZE
            ),
        )
    )

    reconstructed_neighbor_indices = (
        nearest_neighbor_indices(
            representations=(
                standardized_reconstructed
            ),
            number_of_neighbors=(
                NUMBER_OF_STRUCTURE_NEIGHBORS
            ),
            distance_batch_size=(
                DISTANCE_BATCH_SIZE
            ),
        )
    )

    clean_noisy_overlaps = (
        neighborhood_overlap_per_sample(
            reference_neighbor_indices=(
                clean_neighbor_indices
            ),
            comparison_neighbor_indices=(
                noisy_neighbor_indices
            ),
        )
    )

    clean_reconstructed_overlaps = (
        neighborhood_overlap_per_sample(
            reference_neighbor_indices=(
                clean_neighbor_indices
            ),
            comparison_neighbor_indices=(
                reconstructed_neighbor_indices
            ),
        )
    )

    mean_clean_noisy_overlap = (
        clean_noisy_overlaps.mean().item()
    )

    mean_clean_reconstructed_overlap = (
        clean_reconstructed_overlaps.mean().item()
    )

    improved_overlap_fraction = (
        clean_reconstructed_overlaps
        > clean_noisy_overlaps
    ).to(
        torch.float32
    ).mean().item()

    results = {
        "clean_knn_accuracy": (
            clean_accuracy
        ),
        "noisy_knn_accuracy": (
            noisy_accuracy
        ),
        "reconstructed_knn_accuracy": (
            reconstructed_accuracy
        ),
        "clean_noisy_overlap": (
            mean_clean_noisy_overlap
        ),
        "clean_reconstructed_overlap": (
            mean_clean_reconstructed_overlap
        ),
        "improved_overlap_fraction": (
            improved_overlap_fraction
        ),
    }

    print()
    print(model_name)

    print("  k-NN classification accuracy")

    print(
        "    Clean:",
        f"{100.0 * clean_accuracy:.2f}%",
    )

    print(
        "    Noisy:",
        f"{100.0 * noisy_accuracy:.2f}%",
    )

    print(
        "    Reconstructed:",
        f"{100.0 * reconstructed_accuracy:.2f}%",
    )

    print("  Neighborhood preservation")

    print(
        "    Clean-noisy overlap:",
        (
            f"{100.0 * mean_clean_noisy_overlap:.2f}%"
        ),
    )

    print(
        "    Clean-reconstructed overlap:",
        (
            f"{100.0 * mean_clean_reconstructed_overlap:.2f}%"
        ),
    )

    print(
        "    Images with improved overlap:",
        (
            f"{100.0 * improved_overlap_fraction:.2f}%"
        ),
    )

    return results


def save_results_csv(
    all_results: dict[
        str,
        dict[str, float],
    ],
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "latent_structure_metrics.csv"
    )

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model",
                "metric",
                "value",
            ],
        )

        writer.writeheader()

        for model_name, results in (
            all_results.items()
        ):
            for metric_name, value in (
                results.items()
            ):
                writer.writerow(
                    {
                        "model": model_name,
                        "metric": metric_name,
                        "value": value,
                    }
                )

    print()
    print(
        "Results saved in:",
        output_path,
    )


def save_results_figure(
    all_results: dict[
        str,
        dict[str, float],
    ],
) -> None:
    model_names = list(
        all_results.keys()
    )

    accuracy_conditions = [
        "Clean",
        "Noisy",
        "Reconstructed",
    ]

    accuracy_metric_names = [
        "clean_knn_accuracy",
        "noisy_knn_accuracy",
        "reconstructed_knn_accuracy",
    ]

    overlap_conditions = [
        "Clean vs noisy",
        "Clean vs reconstructed",
    ]

    overlap_metric_names = [
        "clean_noisy_overlap",
        "clean_reconstructed_overlap",
    ]

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
    )

    accuracy_positions = np.arange(
        len(accuracy_conditions)
    )

    overlap_positions = np.arange(
        len(overlap_conditions)
    )

    number_of_models = len(
        model_names
    )

    bar_width = 0.8 / number_of_models

    for model_index, model_name in enumerate(
        model_names
    ):
        horizontal_offset = (
            model_index
            - (number_of_models - 1) / 2
        ) * bar_width

        accuracy_values = [
            all_results[
                model_name
            ][metric_name]
            for metric_name in (
                accuracy_metric_names
            )
        ]

        axes[0].bar(
            accuracy_positions
            + horizontal_offset,
            accuracy_values,
            width=bar_width,
            label=model_name,
        )

        overlap_values = [
            all_results[
                model_name
            ][metric_name]
            for metric_name in (
                overlap_metric_names
            )
        ]

        axes[1].bar(
            overlap_positions
            + horizontal_offset,
            overlap_values,
            width=bar_width,
            label=model_name,
        )

    axes[0].set_xticks(
        accuracy_positions
    )

    axes[0].set_xticklabels(
        accuracy_conditions
    )

    axes[0].set_ylabel(
        "Classification accuracy"
    )

    axes[0].set_title(
        "k-NN classification in latent space"
    )

    axes[0].set_ylim(
        0.0,
        1.0,
    )

    axes[0].yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0
        )
    )

    axes[0].grid(
        axis="y"
    )

    axes[0].legend()

    axes[1].set_xticks(
        overlap_positions
    )

    axes[1].set_xticklabels(
        overlap_conditions
    )

    axes[1].set_ylabel(
        "Mean neighborhood overlap"
    )

    axes[1].set_title(
        "Preservation of local latent geometry"
    )

    axes[1].set_ylim(
        0.0,
        1.0,
    )

    axes[1].yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0
        )
    )

    axes[1].grid(
        axis="y"
    )

    axes[1].legend()

    figure.suptitle(
        "Quantitative analysis of latent representations"
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.95)
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURE_DIR
        / "latent_structure_metrics.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.show()

    print(
        "Figure saved in:",
        output_path,
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

    complete_training_dataset = (
        datasets.FashionMNIST(
            root=DATA_DIR,
            train=True,
            download=True,
            transform=transform,
        )
    )

    complete_test_dataset = (
        datasets.FashionMNIST(
            root=DATA_DIR,
            train=False,
            download=True,
            transform=transform,
        )
    )

    reference_dataset = create_random_subset(
        dataset=complete_training_dataset,
        number_of_samples=(
            NUMBER_OF_REFERENCE_IMAGES
        ),
        seed=ANALYSIS_SEED,
    )

    query_dataset = create_random_subset(
        dataset=complete_test_dataset,
        number_of_samples=(
            NUMBER_OF_QUERY_IMAGES
        ),
        seed=ANALYSIS_SEED + 1,
    )

    reference_loader = DataLoader(
        reference_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    fully_connected_model = (
        FullyConnectedDenoisingAutoencoder(
            latent_dim=LATENT_DIM,
        ).to(device)
    )

    convolutional_model = (
        ConvolutionalDenoisingAutoencoder(
            latent_dim=LATENT_DIM,
        ).to(device)
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

    all_results = {}

    all_results[
        "FC autoencoder"
    ] = analyze_model(
        model_name="FC autoencoder",
        model=fully_connected_model,
        reference_loader=reference_loader,
        query_loader=query_loader,
        device=device,
    )

    all_results[
        "Conv autoencoder"
    ] = analyze_model(
        model_name="Conv autoencoder",
        model=convolutional_model,
        reference_loader=reference_loader,
        query_loader=query_loader,
        device=device,
    )

    save_results_csv(
        all_results
    )

    save_results_figure(
        all_results
    )


if __name__ == "__main__":
    main()
