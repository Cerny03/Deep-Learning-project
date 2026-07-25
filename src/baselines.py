import torch
import torch.nn.functional as F 


def gaussian_blur_3x3(images: torch.Tensor) -> torch.Tensor:
    """
    Apply a fixed 3x3 Gaussian-like filter to grayscale images.

    Parameters
    ----------
    images:
        Tensor with shape [B, 1, H, W].

    Returns
    -------
    filtered_images:
        Filtered tensor with the same shape as the input.
    """
    if images.ndim != 4:
        raise ValueError(
            "images must have shape [B, C, H, W]"
        )

    if images.shape[1] != 1:
        raise ValueError(
            "This first implementation supports grayscale images only"
        )

    kernel = torch.tensor(
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=images.dtype,
        device=images.device,
    )

    kernel = kernel / kernel.sum()

    kernel = kernel.view(1, 1, 3, 3) 

    padded_images = F.pad(
        images,
        pad=(1, 1, 1, 1),
        mode="reflect",
    )

    filtered_images = F.conv2d(
        padded_images,
        weight=kernel,
    )

    return filtered_images