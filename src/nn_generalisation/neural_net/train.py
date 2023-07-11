from .neural_net import DenseNN

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

def train(model : DenseNN,
          train_loader : DataLoader,
          optimizer : torch.optim.SGD, 
          args : dict) -> float:
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if not args["pre_transfer"]:
            data, target = data.to(args["device"]), target.to(args["device"])
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target, reduction="sum") / len(data)
        epoch_loss += F.mse_loss(output, target, reduction="sum")
        loss.backward()
        optimizer.step()
    return epoch_loss.item() / len(train_loader.dataset)
