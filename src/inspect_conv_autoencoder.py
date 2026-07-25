from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import ConvolutionalDenoisingAutoencoder
from noise import add_gaussian_noise


BATCH_SIZE = 32
LATENT_DIM = 16
NOISE_SIGMA = 0.3

DATA_DIR = Path("data")

def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


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

    clean_images, _ = next(iter(test_loader))

    noisy_images, _ = add_gaussian_noise(
        clean_images,
        sigma=NOISE_SIGMA,
    )

    model = ConvolutionalDenoisingAutoencoder(
        latent_dim=LATENT_DIM,
    )

    with torch.no_grad():
        encoder_feature_maps = (
            model.convolutional_encoder(noisy_images)
        )

        latent_representations = model.latent_projection(
            encoder_feature_maps
        )

        decoder_feature_maps = model.decoder_projection(
            latent_representations
        )

        reconstructed_images = (
            model.convolutional_decoder(
                decoder_feature_maps
            )
        )

        complete_output = model(noisy_images)

    print("Model")
    print(model)

    print()
    print("Tensor shapes")
    print("  Clean images:", clean_images.shape)
    print("  Noisy images:", noisy_images.shape)

    print(
        "  Encoder feature maps:",
        encoder_feature_maps.shape,
    )

    print(
        "  Latent representations:",
        latent_representations.shape,
    )

    print(
        "  Decoder feature maps:",
        decoder_feature_maps.shape,
    )

    print(
        "  Reconstructed images:",
        reconstructed_images.shape,
    )

    print()
    print("Output range")
    print(
        "  Minimum:",
        reconstructed_images.min().item(),
    )
    print(
        "  Maximum:",
        reconstructed_images.max().item(),
    )

    print()
    print("Model parameters")
    print(
        "  Trainable parameters:",
        count_parameters(model),
    )

    print()
    print("Consistency check")
    print(
        "  Maximum difference:",
        (
            reconstructed_images - complete_output
        ).abs().max().item(),
    )

    assert encoder_feature_maps.shape == (
        BATCH_SIZE,
        32,
        7,
        7,
    )

    assert latent_representations.shape == (
        BATCH_SIZE,
        LATENT_DIM,
    )

    assert decoder_feature_maps.shape == (
        BATCH_SIZE,
        32,
        7,
        7,
    )

    assert reconstructed_images.shape == (
        BATCH_SIZE,
        1,
        28,
        28,
    )

    assert reconstructed_images.min().item() >= 0.0
    assert reconstructed_images.max().item() <= 1.0

    assert torch.allclose(
        reconstructed_images,
        complete_output,
    )


if __name__ == "__main__":
    main()