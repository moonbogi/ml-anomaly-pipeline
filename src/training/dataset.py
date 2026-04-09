"""
Dataset loader for anomaly detection training.
Uses MNIST as a proxy for industrial inspection data:
  - "normal" class = one digit (e.g. 0)
  - everything else = anomalies at inference time

In production (e.g. DarkVision), swap this for your labeled inspection images.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms(img_size: int = 32) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def get_normal_dataloader(
    data_dir: str = "./data/raw",
    normal_class: int = 0,
    img_size: int = 32,
    batch_size: int = 64,
    train: bool = True,
) -> DataLoader:
    """
    Returns a DataLoader containing only the 'normal' class samples.
    The autoencoder trains exclusively on normal samples so it learns
    to reconstruct them well — anomalies produce high reconstruction error.
    """
    transform = get_transforms(img_size)
    dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)

    normal_indices = [i for i, (_, label) in enumerate(dataset) if label == normal_class]
    normal_subset = Subset(dataset, normal_indices)

    return DataLoader(normal_subset, batch_size=batch_size, shuffle=train, num_workers=2)


def get_test_dataloader(
    data_dir: str = "./data/raw",
    img_size: int = 32,
    batch_size: int = 64,
) -> DataLoader:
    """Full test set — used to evaluate anomaly detection (normal vs all others)."""
    transform = get_transforms(img_size)
    dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
