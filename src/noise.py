import torch


def add_gaussian_noise(
    images: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Add independent Gaussian noise to a tensor of images.

    Parameters
    ----------
    images:
        Tensor containing images, usually with shape [B, C, H, W].
        Pixel values are expected to be in the interval [0, 1].

    sigma:
        Standard deviation of the Gaussian noise.
        It must be non-negative.

    Returns
    -------
    noisy_images:
        Corrupted images, clipped to the interval [0, 1].

    noise:
        The Gaussian noise sampled before clipping.
        It has the same shape, dtype, and device as images.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    noise = sigma * torch.randn_like(images)
    noisy_images = torch.clamp(images + noise, min=0.0, max=1.0)

    return noisy_images, noise

def add_salt_and_pepper_noise(
    images: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    """
    Replace a fraction of the pixels with either 0 or 1.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "probability must lie in [0, 1]"
        )

    random_values = torch.rand_like(images)

    noisy_images = images.clone()

    pepper_mask = random_values < (
        probability / 2.0
    )

    salt_mask = (
        random_values >= probability / 2.0
    ) & (
        random_values < probability
    )

    noisy_images[pepper_mask] = 0.0
    noisy_images[salt_mask] = 1.0

    return noisy_images


def add_random_masking_noise(
    images: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    """
    Replace a random fraction of the pixels with zero.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "probability must lie in [0, 1]"
        )

    mask = torch.rand_like(images) < probability

    noisy_images = images.clone()
    noisy_images[mask] = 0.0

    return noisy_images