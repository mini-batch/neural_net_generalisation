import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class DenseNN(nn.Module):
    """
    Fully connected neural network
    """
    def __init__(self, num_hidden_units):
        super(DenseNN, self).__init__()
        self.num_hidden_units = num_hidden_units
        self.l1 = nn.Linear(784, num_hidden_units)
        self.activation_fun = nn.ReLU()
        self.l2 = nn.Linear(num_hidden_units, 10)

    def forward(self, x):
        return self.l2(self.activation_fun(self.l1(x)))
    
    def set_params(self, params : list[torch.Tensor]):
        """
        Set the parameters of the neural network to the given values
        params:
            params: list of Tensors which match the sizes of the current models parameters
        """
        ind = 0
        for p in self.parameters():
            if p.requires_grad:
                if p.shape != params[ind].shape:
                    raise Exception(f"Supplied parameters did not match model parameter shapes: parameter group {ind} had shape {params[0].shape} but expected shape {p.shape}")
                p.data = params[ind].data
                ind += 1

    def get_params(self) -> list[torch.Tensor]:
        """
        Return a list of the current model parameters
        """
        return [p for p in self.parameters() if p.requires_grad]


def train(model : DenseNN,
          train_loader : DataLoader,
          optimizer : torch.optim.SGD,
          loss_fn : torch.nn.modules.loss._Loss | torch.nn.modules.loss._WeightedLoss,
          args : dict) -> float:
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if not args["pre_transfer"]:
            data, target = data.to(args["device"]), target.to(args["device"])
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target) / len(data)
        epoch_loss += loss * len(data)
        loss.backward()
        optimizer.step()
    return epoch_loss.item() / len(train_loader.dataset)


def test(model : DenseNN,
        test_loader : DataLoader,
        loss_fn : torch.nn.modules.loss._Loss | torch.nn.modules.loss._WeightedLoss,
        args : dict) -> float:
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            if not args["pre_transfer"]:
                data, target = data.to(args["device"]), target.to(args["device"])
            output = model(data)
            test_loss += loss_fn(output, target)
    return test_loss.item() / len(test_loader.dataset)


def extend_params(params : list[torch.Tensor],
                  new_model_size : int,
                  device : torch.device) -> list[torch.Tensor]:
    """
    Extend parameters to shape of larger model, new parameters are set to 0.
    """
    hidden_unit_diff = new_model_size - params[0].size(dim=0)
    params[0] = torch.cat([params[0], torch.normal(0, 0.01, size=(hidden_unit_diff, 784), device=device)], dim=0)
    assert params[0].shape == (new_model_size, 784), f"Expected {(new_model_size, 784)} but got {params[0].shape}"
    params[1] = torch.cat([params[1], torch.normal(0, 0.01, size=(hidden_unit_diff,), device=device)], dim=0)
    assert params[1].shape[0] == (new_model_size), f"Expected {new_model_size} but got {params[1].shape}"
    params[2] = torch.cat([params[2], torch.normal(0, 0.1, size=(10, hidden_unit_diff), device=device)], dim=1)
    assert params[2].shape == (10, new_model_size), f"Expected {(10, new_model_size)} but got {params[2].shape}"
    return params
