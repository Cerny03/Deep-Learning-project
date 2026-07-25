import torch
from torch import nn

class FullyConnectedDenoisingAutoencoder(nn.Module):
    """
    Fully connected denoising autoencoder for 28x28 grayscale images.
    """

    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                in_features=28 * 28,
                out_features=256,
            ),

            nn.ReLU(),

            nn.Linear(
                in_features=256,
                out_features=64,
            ),
            nn.ReLU(),

            nn.Linear(
                in_features=64,
                out_features=latent_dim,
            ),
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=latent_dim,
                out_features=64,
            ),
            nn.ReLU(),

            nn.Linear(
                in_features=64,
                out_features=256,
            ),
            nn.ReLU(),

            nn.Linear(
                in_features=256,
                out_features=28 * 28,
            ),
            nn.Sigmoid(),
            
            nn.Unflatten(
                dim=1,
                unflattened_size=(1, 28, 28),
            ),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images into latent representations.
        """
        return self.encoder(images)

    def decode(
        self,
        latent_representations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert latent representations into reconstructed images.
        """
        return self.decoder(latent_representations)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Perform the complete encoder-decoder transformation.
        """
        latent_representations = self.encode(images)
        reconstructed_images = self.decode(latent_representations)

        return reconstructed_images
    

class ConvolutionalDenoisingAutoencoder(nn.Module):
    """
    Convolutional denoising autoencoder for 28x28 grayscale images.
    """

    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.latent_dim = latent_dim

        self.convolutional_encoder = nn.Sequential(
            #[B, 1, 28, 28] → [B, 16, 14, 14] → [B, 32, 7, 7]
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.latent_projection = nn.Sequential(
            #[B, 32, 7, 7] → [B, 1568]
            nn.Flatten(),

            nn.Linear(
                #[B, 1568] → [B, 16] bottleneck
                in_features=32 * 7 * 7,
                out_features=latent_dim,
            ),
        )

        self.decoder_projection = nn.Sequential(
            
            nn.Linear(
                in_features=latent_dim,
                out_features=32 * 7 * 7,
            ),
            nn.ReLU(),

            nn.Unflatten(
                dim=1,
                unflattened_size=(32, 7, 7),
            ),
        )

        self.convolutional_decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        feature_maps = self.convolutional_encoder(images)

        latent_representations = self.latent_projection(
            feature_maps
        )

        return latent_representations

    def decode(
        self,
        latent_representations: torch.Tensor,
    ) -> torch.Tensor:
        feature_maps = self.decoder_projection(
            latent_representations
        )

        reconstructed_images = self.convolutional_decoder(
            feature_maps
        )

        return reconstructed_images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        latent_representations = self.encode(images)

        reconstructed_images = self.decode(
            latent_representations
        )

        return reconstructed_images