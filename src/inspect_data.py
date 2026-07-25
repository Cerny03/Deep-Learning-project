from pathlib import Path 
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 

BATCH_SIZE = 32 
DATA_DIR = Path("data") 
FIGURE_DIR = Path("figures")


def main():
    transform = transforms.ToTensor() 

    train_dataset = datasets.FashionMNIST( 
        root=DATA_DIR, 
        train=True, 
        download=True, 
        transform=transform, 
    )

    test_dataset = datasets.FashionMNIST( 
        root=DATA_DIR,
        train=False,  
        download=True,
        transform=transform,
    )

    train_loader = DataLoader( 
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0, 
    )

    image, label = train_dataset[0] 

    print("Single image")
    print("  Shape:", image.shape)
    print("  Data type:", image.dtype)
    print("  Label:", label)
    print("  Class:", train_dataset.classes[label])
    print("  Minimum pixel value:", image.min().item())
    print("  Maximum pixel value:", image.max().item())

    print()
    print("Dataset sizes")
    print("  Training samples:", len(train_dataset))
    print("  Test samples:", len(test_dataset))

    images, labels = next(iter(train_loader)) 

    print()
    print("Mini-batch")
    print("  Images shape:", images.shape)
    print("  Labels shape:", labels.shape)

    FIGURE_DIR.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(10, 6))

    for axis, current_image, current_label in zip(
        axes.flatten(), 
        images[:8],
        labels[:8],
    ):
        axis.imshow(current_image.squeeze(0), cmap="gray") 
        axis.set_title(train_dataset.classes[current_label.item()])
        axis.axis("off")

    figure.suptitle("Fashion-MNIST training samples")
    figure.tight_layout()

    output_path = FIGURE_DIR / "fashion_mnist_samples.png"
    figure.savefig(output_path, dpi=150) 
    plt.show()

    print()
    print("Figure saved in:", output_path)

if __name__ == "__main__":
    main()
