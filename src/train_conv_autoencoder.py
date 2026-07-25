from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models import ConvolutionalDenoisingAutoencoder
from noise import add_gaussian_noise


BATCH_SIZE = 32
LATENT_DIM = 16
NOISE_SIGMA = 0.3

LEARNING_RATE = 1e-3
NUMBER_OF_EPOCHS = 10
VALIDATION_FRACTION = 0.1

NUMBER_OF_IMAGES = 6

DATA_DIR = Path("data")
FIGURE_DIR = Path("figures")
CHECKPOINT_DIR = Path("checkpoints")


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for clean_images, _ in data_loader:
        clean_images = clean_images.to(device)

        noisy_images, _ = add_gaussian_noise(
            clean_images,
            sigma=NOISE_SIGMA,
        )

        optimizer.zero_grad()

        reconstructed_images = model(noisy_images)

        loss = loss_function(
            reconstructed_images,
            clean_images,
        )

        loss.backward()
        optimizer.step()

        current_batch_size = clean_images.shape[0]

        total_loss += loss.item() * current_batch_size
        total_samples += current_batch_size

    average_loss = total_loss / total_samples

    return average_loss


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for clean_images, _ in data_loader:
            clean_images = clean_images.to(device)

            noisy_images, _ = add_gaussian_noise(
                clean_images,
                sigma=NOISE_SIGMA,
            )

            reconstructed_images = model(noisy_images)

            loss = loss_function(
                reconstructed_images,
                clean_images,
            )

            current_batch_size = clean_images.shape[0]

            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

    average_loss = total_loss / total_samples

    return average_loss


def save_loss_curve(
    training_losses: list[float],
    validation_losses: list[float],
) -> None:
    epochs = range(1, len(training_losses) + 1)

    figure, axis = plt.subplots(figsize=(7, 5))

    axis.plot(
        epochs,
        training_losses,
        marker="o",
        label="Training loss",
    )

    axis.plot(
        epochs,
        validation_losses,
        marker="o",
        label="Validation loss",
    )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean squared error")
    axis.set_title("Convolutional denoising autoencoder")
    axis.legend()
    axis.grid(True)

    figure.tight_layout()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURE_DIR
        / "conv_autoencoder_loss.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.show()

    print("Loss curve saved in:", output_path)


def save_reconstruction_figure(
    model: nn.Module,
    data_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> None:
    model.eval()

    clean_images, labels = next(iter(data_loader))
    clean_images = clean_images.to(device)

    noisy_images, _ = add_gaussian_noise(
        clean_images,
        sigma=NOISE_SIGMA,
    )

    with torch.no_grad():
        reconstructed_images = model(noisy_images)

    clean_images = clean_images.cpu()
    noisy_images = noisy_images.cpu()
    reconstructed_images = reconstructed_images.cpu()

    figure, axes = plt.subplots(
        3,
        NUMBER_OF_IMAGES,
        figsize=(12, 6),
    )

    image_groups = [
        ("Clean", clean_images),
        ("Noisy", noisy_images),
        (
            "Conv autoencoder",
            reconstructed_images,
        ),
    ]

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
            fontsize=11,
        )

    figure.suptitle(
        "Convolutional autoencoder, "
        f"noise sigma = {NOISE_SIGMA}"
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.95),
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURE_DIR
        / "conv_autoencoder_reconstructions.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.show()

    print(
        "Reconstruction figure saved in:",
        output_path,
    )


def main() -> None:
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    transform = transforms.ToTensor()

    complete_training_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    validation_size = int(
        VALIDATION_FRACTION
        * len(complete_training_dataset)
    )

    training_size = (
        len(complete_training_dataset)
        - validation_size
    )

    split_generator = torch.Generator().manual_seed(42)

    training_dataset, validation_dataset = random_split(
        complete_training_dataset,
        lengths=[
            training_size,
            validation_size,
        ],
        generator=split_generator,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        "Training samples:",
        len(training_dataset),
    )

    print(
        "Validation samples:",
        len(validation_dataset),
    )

    model = ConvolutionalDenoisingAutoencoder(
        latent_dim=LATENT_DIM,
    )

    model = model.to(device)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / "best_conv_autoencoder.pt"
    )

    training_losses = []
    validation_losses = []

    best_validation_loss = float("inf")

    for epoch in range(1, NUMBER_OF_EPOCHS + 1):
        training_loss = train_one_epoch(
            model=model,
            data_loader=training_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        validation_loss = evaluate(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)

        print(
            f"Epoch {epoch:02d}/{NUMBER_OF_EPOCHS} "
            f"| training loss: {training_loss:.6f} "
            f"| validation loss: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print("  New best model saved.")

    print()
    print(
        "Best validation loss:",
        best_validation_loss,
    )

    print(
        "Best model saved in:",
        checkpoint_path,
    )

    best_state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(best_state_dict)

    save_loss_curve(
        training_losses=training_losses,
        validation_losses=validation_losses,
    )

    save_reconstruction_figure(
        model=model,
        data_loader=validation_loader,
        class_names=complete_training_dataset.classes,
        device=device,
    )


if __name__ == "__main__":
    main()
