import torch
from torch.utils.data import DataLoader, TensorDataset


def one_hot_transform(y):
        """
        Transform to convert class labels to one-hot representation
        params:
            y: [list : int]
        return (tensor, n x num_classes)
        """
        return torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)


def load_data_to_device(train_dataset, test_dataset, device):
    """
    Given a suitably sized dataset, transfer to device before training and return related DataLoader objects.
    As MNIST is small, saves data transfer on every train/test iteration.
    params:
        train_dataset:
        test_dataset:
        device:
        train_kwargs:
        test_kwargs:
    return: (DataLoader, Dataloader)
    """
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    _, (images, labels) = next(enumerate(train_loader))
    images, labels = images.to(device), labels.to(device)
    train_loader = DataLoader(TensorDataset(images, labels))
    
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    _, (images, labels) = next(enumerate(test_loader))
    images, labels = images.to(device), labels.to(device)
    test_loader = DataLoader(TensorDataset(images, labels))
    
    return train_loader, test_loader
