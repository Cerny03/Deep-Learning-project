from collections.abc import Callable
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader


def collect_latent_representations(
    model: nn.Module,
    data_loader: DataLoader,
    corruption_function: Callable[
        [torch.Tensor],
        torch.Tensor,
    ],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Collect latent representations of clean, noisy,
    and reconstructed images.
    The returned dictionary contains:
        clean:
            Latent representation of each clean image.
        noisy:
            Latent representation of the corresponding
            corrupted image.
        reconstructed:
            Latent representation obtained by encoding
            the reconstructed image again.
        labels:
            Fashion-MNIST class labels.
    """
    model.eval()

    clean_latent_batches = []
    noisy_latent_batches = []
    reconstructed_latent_batches = []
    label_batches = []

    with torch.no_grad():
        for clean_images, labels in data_loader:
            clean_images = clean_images.to(device)

            noisy_images = corruption_function(
                clean_images
            )

            reconstructed_images = model(
                noisy_images
            )

            clean_latent = model.encode(
                clean_images
            )

            noisy_latent = model.encode(
                noisy_images
            )

            reconstructed_latent = model.encode(
                reconstructed_images
            )

            clean_latent_batches.append(
                clean_latent.cpu()
            )

            noisy_latent_batches.append(
                noisy_latent.cpu()
            )

            reconstructed_latent_batches.append(
                reconstructed_latent.cpu()
            )

            label_batches.append(
                labels.cpu()
            )

    clean_latent_representations = torch.cat(
        clean_latent_batches,
        dim=0,
    )

    noisy_latent_representations = torch.cat(
        noisy_latent_batches,
        dim=0,
    )

    reconstructed_latent_representations = torch.cat(
        reconstructed_latent_batches,
        dim=0,
    )

    all_labels = torch.cat(
        label_batches,
        dim=0,
    )

    return {
        "clean": clean_latent_representations,
        "noisy": noisy_latent_representations,
        "reconstructed": (
            reconstructed_latent_representations
        ),
        "labels": all_labels,
    }


def compute_latent_recovery_metrics(
    representations: dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    Compare noisy and reconstructed latent representations
    with the corresponding clean latent representations.
    """
    clean_latent = representations["clean"]
    noisy_latent = representations["noisy"]

    reconstructed_latent = representations[
        "reconstructed"
    ]

    if clean_latent.shape != noisy_latent.shape:
        raise ValueError(
            "Clean and noisy latent representations "
            "must have the same shape"
        )

    if clean_latent.shape != reconstructed_latent.shape:
        raise ValueError(
            "Clean and reconstructed latent representations "
            "must have the same shape"
        )

    clean_noisy_differences = (
        noisy_latent - clean_latent
    )

    clean_reconstructed_differences = (
        reconstructed_latent - clean_latent
    )

    noisy_distances = torch.linalg.vector_norm(
        clean_noisy_differences,
        ord=2,
        dim=1,
    )

    reconstructed_distances = (
        torch.linalg.vector_norm(
            clean_reconstructed_differences,
            ord=2,
            dim=1,
        )
    )

    mean_noisy_distance = (
        noisy_distances.mean().item()
    )

    mean_reconstructed_distance = (
        reconstructed_distances.mean().item()
    )

    if mean_noisy_distance == 0.0:
        relative_recovery = 0.0
    else:
        relative_recovery = (
            1.0
            - mean_reconstructed_distance
            / mean_noisy_distance
        )

    improved_images = (
        reconstructed_distances
        < noisy_distances
    )

    improved_fraction = (
        improved_images
        .to(torch.float32)
        .mean()
        .item()
    )

    return {
        "clean_noisy_distance": (
            mean_noisy_distance
        ),
        "clean_reconstructed_distance": (
            mean_reconstructed_distance
        ),
        "relative_recovery": (
            relative_recovery
        ),
        "improved_fraction": (
            improved_fraction
        ),
    }


def shared_pca_projection(
    clean_representations: torch.Tensor,
    noisy_representations: torch.Tensor,
    reconstructed_representations: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Project clean, noisy, and reconstructed representations
    onto the same two-dimensional PCA plane.

    PCA is fitted jointly on the three groups. This is necessary
    so that the three resulting plots use the same coordinate system.
    """
    if (
        clean_representations.ndim != 2
        or noisy_representations.ndim != 2
        or reconstructed_representations.ndim != 2
    ):
        raise ValueError(
            "All representations must have shape "
            "[number_of_samples, latent_dimension]"
        )

    if not (
        clean_representations.shape
        == noisy_representations.shape
        == reconstructed_representations.shape
    ):
        raise ValueError(
            "Clean, noisy, and reconstructed representations "
            "must have the same shape"
        )

    number_of_samples = (
        clean_representations.shape[0]
    )

    combined_representations = torch.cat(
        [
            clean_representations,
            noisy_representations,
            reconstructed_representations,
        ],
        dim=0,
    )

    representation_mean = (
        combined_representations.mean(
            dim=0,
            keepdim=True,
        )
    )

    centered_representations = (
        combined_representations
        - representation_mean
    )

    _, _, principal_directions = (
        torch.pca_lowrank(
            centered_representations,
            q=2,
            center=False,
        )
    )

    projected_representations = (
        centered_representations
        @ principal_directions[:, :2]
    )

    clean_projection = (
        projected_representations[
            :number_of_samples
        ]
    )

    noisy_projection = (
        projected_representations[
            number_of_samples:
            2 * number_of_samples
        ]
    )

    reconstructed_projection = (
        projected_representations[
            2 * number_of_samples:
        ]
    )

    return (
        clean_projection,
        noisy_projection,
        reconstructed_projection,
    )


def save_latent_pca_figure(
    clean_projection: torch.Tensor,
    noisy_projection: torch.Tensor,
    reconstructed_projection: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    """
    Save three PCA scatter plots:
        1. clean representations;
        2. noisy representations;
        3. reconstructed representations.
    """
    if clean_projection.shape[1] != 2:
        raise ValueError(
            "PCA projections must have two columns"
        )

    projection_groups = [
        (
            "Clean input",
            clean_projection,
        ),
        (
            "Noisy input",
            noisy_projection,
        ),
        (
            "Reconstructed input",
            reconstructed_projection,
        ),
    ]

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 4.5),
        sharex=True,
        sharey=True,
    )

    scatter_plot = None

    for axis, (
        projection_name,
        projection,
    ) in zip(
        axes,
        projection_groups,
    ):
        scatter_plot = axis.scatter(
            projection[:, 0],
            projection[:, 1],
            c=labels,
            cmap="tab10",
            vmin=-0.5,
            vmax=9.5,
            s=7,
            alpha=0.55,
        )

        axis.set_title(
            projection_name
        )

        axis.set_xlabel(
            "First principal component"
        )

        axis.grid(True)

    axes[0].set_ylabel(
        "Second principal component"
    )

    if scatter_plot is None:
        raise RuntimeError(
            "No PCA points were plotted"
        )

    colorbar = figure.colorbar(
        scatter_plot,
        ax=axes,
        ticks=range(len(class_names)),
        fraction=0.025,
        pad=0.02,
    )

    colorbar.ax.set_yticklabels(
        class_names
    )

    figure.suptitle(
        title
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()

    print(
        "Figure saved in:",
        output_path,
    )


def collect_clean_latent_representations(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collect latent representations of clean images and
    their corresponding labels.
    """
    model.eval()

    latent_batches = []
    label_batches = []

    with torch.no_grad():
        for clean_images, labels in data_loader:
            clean_images = clean_images.to(device)

            clean_latent = model.encode(
                clean_images
            )

            latent_batches.append(
                clean_latent.cpu()
            )

            label_batches.append(
                labels.cpu()
            )

    latent_representations = torch.cat(
        latent_batches,
        dim=0,
    )

    labels = torch.cat(
        label_batches,
        dim=0,
    )

    return latent_representations, labels


def standardize_using_reference(
    reference_representations: torch.Tensor,
    *other_representations: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """
    Standardize latent dimensions using the mean and
    standard deviation of the clean reference set.

    The same transformation is then applied to every
    other representation set.
    """
    if reference_representations.ndim != 2:
        raise ValueError(
            "Reference representations must have shape "
            "[number_of_samples, latent_dimension]"
        )

    latent_dimension = (
        reference_representations.shape[1]
    )

    for representations in other_representations:
        if representations.ndim != 2:
            raise ValueError(
                "Every representation tensor must be "
                "two-dimensional"
            )

        if representations.shape[1] != latent_dimension:
            raise ValueError(
                "All representations must have the same "
                "latent dimension"
            )

    reference_mean = (
        reference_representations.mean(
            dim=0,
            keepdim=True,
        )
    )

    reference_standard_deviation = (
        reference_representations.std(
            dim=0,
            unbiased=False,
            keepdim=True,
        )
    )

    reference_standard_deviation = (
        reference_standard_deviation.clamp_min(
            1e-6
        )
    )

    all_representations = (
        reference_representations,
        *other_representations,
    )

    standardized_representations = tuple(
        (
            representations
            - reference_mean
        )
        / reference_standard_deviation
        for representations in all_representations
    )

    return standardized_representations


def knn_classification_accuracy(
    reference_representations: torch.Tensor,
    reference_labels: torch.Tensor,
    query_representations: torch.Tensor,
    query_labels: torch.Tensor,
    number_of_neighbors: int = 5,
    distance_batch_size: int = 256,
) -> float:
    """
    Classify each query representation using the majority
    label among its nearest reference representations.
    """
    if number_of_neighbors <= 0:
        raise ValueError(
            "number_of_neighbors must be positive"
        )

    if (
        number_of_neighbors
        > reference_representations.shape[0]
    ):
        raise ValueError(
            "The number of neighbors cannot exceed "
            "the number of reference samples"
        )

    if (
        reference_representations.shape[0]
        != reference_labels.shape[0]
    ):
        raise ValueError(
            "Reference representations and labels must "
            "contain the same number of samples"
        )

    if (
        query_representations.shape[0]
        != query_labels.shape[0]
    ):
        raise ValueError(
            "Query representations and labels must contain "
            "the same number of samples"
        )

    maximum_label = torch.cat(
        [
            reference_labels,
            query_labels,
        ]
    ).max()

    number_of_classes = (
        int(maximum_label.item()) + 1
    )

    number_of_correct_predictions = 0
    number_of_queries = (
        query_representations.shape[0]
    )

    for start_index in range(
        0,
        number_of_queries,
        distance_batch_size,
    ):
        end_index = min(
            start_index + distance_batch_size,
            number_of_queries,
        )

        query_batch = query_representations[
            start_index:end_index
        ]

        query_label_batch = query_labels[
            start_index:end_index
        ]

        distances = torch.cdist(
            query_batch,
            reference_representations,
            p=2,
        )

        nearest_neighbor_indices = (
            distances.topk(
                k=number_of_neighbors,
                dim=1,
                largest=False,
            ).indices
        )

        nearest_neighbor_labels = (
            reference_labels[
                nearest_neighbor_indices
            ]
        )

        one_hot_neighbor_labels = F.one_hot(
            nearest_neighbor_labels,
            num_classes=number_of_classes,
        )

        class_vote_counts = (
            one_hot_neighbor_labels.sum(
                dim=1
            )
        )

        predicted_labels = (
            class_vote_counts.argmax(
                dim=1
            )
        )

        number_of_correct_predictions += (
            predicted_labels
            == query_label_batch
        ).sum().item()

    accuracy = (
        number_of_correct_predictions
        / number_of_queries
    )

    return accuracy


def nearest_neighbor_indices(
    representations: torch.Tensor,
    number_of_neighbors: int = 10,
    distance_batch_size: int = 256,
) -> torch.Tensor:
    """
    Find the nearest neighbors of every representation
    within the same data set.

    Each sample is explicitly excluded from its own
    neighborhood.
    """
    if representations.ndim != 2:
        raise ValueError(
            "Representations must have shape "
            "[number_of_samples, latent_dimension]"
        )

    number_of_samples = (
        representations.shape[0]
    )

    if not 0 < number_of_neighbors < number_of_samples:
        raise ValueError(
            "number_of_neighbors must be positive and "
            "smaller than the number of samples"
        )

    neighbor_index_batches = []

    for start_index in range(
        0,
        number_of_samples,
        distance_batch_size,
    ):
        end_index = min(
            start_index + distance_batch_size,
            number_of_samples,
        )

        current_batch = representations[
            start_index:end_index
        ]

        distances = torch.cdist(
            current_batch,
            representations,
            p=2,
        )

        local_row_indices = torch.arange(
            end_index - start_index
        )

        global_column_indices = torch.arange(
            start_index,
            end_index,
        )

        distances[
            local_row_indices,
            global_column_indices,
        ] = float("inf")

        nearest_indices = distances.topk(
            k=number_of_neighbors,
            dim=1,
            largest=False,
        ).indices

        neighbor_index_batches.append(
            nearest_indices
        )

    return torch.cat(
        neighbor_index_batches,
        dim=0,
    )


def neighborhood_overlap_per_sample(
    reference_neighbor_indices: torch.Tensor,
    comparison_neighbor_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the fraction of neighbors shared by two
    neighborhood structures for every sample.
    """
    if (
        reference_neighbor_indices.shape
        != comparison_neighbor_indices.shape
    ):
        raise ValueError(
            "The two neighbor-index tensors must have "
            "the same shape"
        )

    number_of_neighbors = (
        reference_neighbor_indices.shape[1]
    )

    pairwise_matches = (
        reference_neighbor_indices.unsqueeze(
            dim=2
        )
        ==
        comparison_neighbor_indices.unsqueeze(
            dim=1
        )
    )

    shared_neighbors = (
        pairwise_matches.any(dim=2).sum(dim=1)
    )

    overlaps = (
        shared_neighbors.to(
            torch.float32
        )
        / number_of_neighbors
    )

    return overlaps