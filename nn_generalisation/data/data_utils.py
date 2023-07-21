import numpy as np
import torch
from torch.utils.data import Subset, DataLoader, TensorDataset
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


def one_hot_transform(y):
        """
        Transform to convert class labels to one-hot representation
        params:
            y: [list : int]
        return (tensor, n x num_classes)
        """
        return torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)


def load_data_to_device(train_dataset, test_dataset, args):
    """
    Given a suitably sized dataset, transfer to device before training and return related DataLoader objects.
    As MNIST is small, saves data transfer on every train/test iteration.
    params:
        train_dataset:
        test_dataset:
        args
    return: (DataLoader, Dataloader)
    """
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    _, (images, labels) = next(enumerate(train_loader))
    images, labels = images.to(args["device"]), labels.to(args["device"])
    train_loader = DataLoader(TensorDataset(images, labels), batch_size=args["batch_size"], shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    _, (images, labels) = next(enumerate(test_loader))
    images, labels = images.to(args["device"]), labels.to(args["device"])
    test_loader = DataLoader(TensorDataset(images, labels), batch_size=args["test_batch_size"], shuffle=True)
    
    return train_loader, test_loader