from .utils import load_data_to_device, one_hot_transform

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms


def get_data(args) -> tuple[DataLoader, DataLoader]:
    """
    Get data loaders for train and test data
    """
    transform=transforms.Compose([
            transforms.ToTensor(),
            torch.flatten
            ])

    data_rng = np.random.RandomState(args["data_seed"])

    dataset1 = datasets.MNIST('./mnist_data', train=True, download=True,
                    transform=transform, target_transform=transforms.Compose([one_hot_transform]))
    dataset2 = datasets.MNIST('./mnist_data', train=False, download=True,
                    transform=transform, target_transform=transforms.Compose([one_hot_transform]))
    dataset1 = Subset(dataset1, data_rng.choice(len(dataset1), args["train_size"], replace=False))

    if not args["pre_transfer"]:
        # For use if not transferring to GPU before training loop
        train_loader = DataLoader(dataset1, batch_size=args["batch_size"], shuffle=True)
        test_loader = DataLoader(dataset2, batch_size=args["test_batch_size"])
    else:
        # Transfer whole datasets to device (i.e. GPU) before training (Faster)
        train_loader, test_loader = load_data_to_device(dataset1, dataset2, args)
    return train_loader, test_loader