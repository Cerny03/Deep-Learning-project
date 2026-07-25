import csv 
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from evaluation_utils import (
    METHOD_NAMES,
    evaluate_corruption,
    load_trained_models,
)
from noise import (
    add_gaussian_noise,
    add_random_masking_noise,
    add_salt_and_pepper_noise,
)


BATCH_SIZE = 32
EVALUATION_SEED = 12345

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def gaussian_corruption(
    images: torch.Tensor,
) -> torch.Tensor:
    noisy_images, _ = add_gaussian_noise(
        images,
        sigma=0.3,
    )

    return noisy_images


def salt_and_pepper_corruption(
    images: torch.Tensor,
) -> torch.Tensor:
    return add_salt_and_pepper_noise(
        images,
        probability=0.2,
    )


def masking_corruption(
    images: torch.Tensor,
) -> torch.Tensor:
    return add_random_masking_noise(
        images,
        probability=0.3,
    )


def print_results(
    corruption_name: str,
    results: dict[str, dict[str, float]],
) -> None:
    print()
    print(corruption_name)

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


def save_results(
    all_results: dict[
        str,
        dict[str, dict[str, float]],
    ],
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / "unseen_corruptions.csv"
    )

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "corruption",
                "method",
                "mse",
                "psnr_db",
            ],
        )

        writer.writeheader()

        for corruption_name, results in all_results.items():
            for method_name, metrics in results.items():
                writer.writerow(
                    {
                        "corruption": corruption_name,
                        "method": method_name,
                        "mse": metrics["mse"],
                        "psnr_db": metrics["psnr"],
                    }
                )

    print()
    print("Results saved in:", output_path)


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    fc_model, conv_model = load_trained_models(
        device=device
    )

    corruption_functions = {
        "Gaussian sigma = 0.3": gaussian_corruption,
        "Salt-and-pepper probability = 0.2": (
            salt_and_pepper_corruption
        ),
        "Random masking probability = 0.3": (
            masking_corruption
        ),
    }

    all_results = {}

    for corruption_name, corruption_function in (
        corruption_functions.items()
    ):
        torch.manual_seed(EVALUATION_SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                EVALUATION_SEED
            )

        results = evaluate_corruption(
            test_loader=test_loader,
            fc_model=fc_model,
            conv_model=conv_model,
            corruption_function=corruption_function,
            device=device,
        )

        all_results[corruption_name] = results

        print_results(
            corruption_name=corruption_name,
            results=results,
        )

    save_results(all_results)


if __name__ == "__main__":
    main()