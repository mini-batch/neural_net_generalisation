from .neural_net import DenseNN

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

def test(model : DenseNN,
        test_loader : DataLoader,
        args : dict) -> float:
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            if not args["pre_transfer"]:
                data, target = data.to(args["device"]), target.to(args["device"])
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction="sum").item()
    return test_loss / len(test_loader.dataset)