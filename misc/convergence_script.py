import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch import optim
from tqdm import tqdm, trange
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader, TensorDataset
from torchvision import datasets, transforms
from datetime import datetime
import json

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
        model.zero_grad()
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
    model.zero_grad(set_to_none=False)
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            if not args["pre_transfer"]:
                data, target = data.to(args["device"]), target.to(args["device"])
            output = model(data)
            test_loss += loss_fn(output, target)
    return test_loss.item() / len(test_loader.dataset)

def cat_and_flatten(input : tuple[torch.Tensor]):
	return torch.cat([torch.flatten(i) for i in input])

def set_seeds(seed : int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_device(args : dict):
    if torch.cuda.is_available():
        args["device"] = torch.device("cuda")
    else:
        args["device"] = torch.device("cpu")
    return args

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
        train_rng = torch.Generator()
        train_rng.manual_seed(args["data_seed"])
        test_rng = torch.Generator()
        train_rng.manual_seed(args["data_seed"] + 1)
        train_loader = DataLoader(dataset1, batch_size=args["batch_size"], shuffle=True, generator=train_rng)
        test_loader = DataLoader(dataset2, batch_size=args["test_batch_size"], generator=test_rng)
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
    train_rng = torch.Generator()
    train_rng.manual_seed(args["data_seed"])
    test_rng = torch.Generator()
    train_rng.manual_seed(args["data_seed"] + 1)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    _, (images, labels) = next(enumerate(train_loader))
    images, labels = images.to(args["device"]), labels.to(args["device"])
    train_loader = DataLoader(TensorDataset(images, labels), batch_size=args["batch_size"], shuffle=True, generator=train_rng)
    
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    _, (images, labels) = next(enumerate(test_loader))
    images, labels = images.to(args["device"]), labels.to(args["device"])
    test_loader = DataLoader(TensorDataset(images, labels), batch_size=args["test_batch_size"], shuffle=True, generator=test_rng)
    return train_loader, test_loader

def setup_log_path(datetime : str) -> str:
    os.makedirs(f"./log/{datetime}/", exist_ok = True)
    return f"./log/{datetime}/"

def setup_log(path : str) -> None:
    logging.basicConfig(filename=path,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG,
                            force=True)
    
def get_datetime_str() -> str:
    date_now = str(datetime.now().date())
    time_now = str(datetime.now().time())[:-7].replace(":", ";")
    return f"{date_now}_{time_now}"

def save_model(model, path : str) -> None:
    torch.save(model.state_dict(), path)

def save_json(obj, path : str) -> None:
    with open(path, "w") as outfile:
        outfile.write(json.dumps(obj, indent=4))

def load_model(num_hidden_units, path : str):
    model = DenseNN(num_hidden_units)
    model.load_state_dict(torch.load(path))
    return model

def load_json(path : str) -> dict:
    with open(path, "r") as infile:
        return json.load(infile)
    
def extend_params(params : list[torch.Tensor],
                  new_model_size : int,
                  device : torch.device) -> list[torch.Tensor]:
    """Extend parameters to shape of larger model, new parameters are set to 0."""
    hidden_unit_diff = new_model_size - params[0].size(dim=0)
    params[0] = torch.cat([params[0], torch.normal(0, 0.01, size=(hidden_unit_diff, 784), device=device)], dim=0)
    assert params[0].shape == (new_model_size, 784), f"Expected {(new_model_size, 784)} but got {params[0].shape}"
    params[1] = torch.cat([params[1], torch.normal(0, 0.01, size=(hidden_unit_diff,), device=device)], dim=0)
    assert params[1].shape[0] == (new_model_size), f"Expected {new_model_size} but got {params[1].shape}"
    params[2] = torch.cat([params[2], torch.normal(0, 0.1, size=(10, hidden_unit_diff), device=device)], dim=1)
    assert params[2].shape == (10, new_model_size), f"Expected {(10, new_model_size)} but got {params[2].shape}"
    return params

def sattr(d, *attrs):
    # Adds "val" to dict in the hierarchy mentioned via *attrs
    for attr in attrs[:-2]:
        # If such key is not found or the value is primitive supply an empty dict
        if d.get(attr) is None or not isinstance(d.get(attr), dict):
            d[attr] = {}
        d = d[attr]
    d[attrs[-2]] = attrs[-1]

def setup_exp(args) -> tuple[dict, dict, str, DataLoader, DataLoader]:
    """Set state of rng's, setup logging, get device, and get data.
    return: experiment_log, args, log_path, train_loader, test_loader
    """
    experiment_log = {"args": args}
    set_seeds(args["seed"])
    args = set_device(args)
    exp_datetime = get_datetime_str()
    log_path = setup_log_path(exp_datetime)
    setup_log(f"{log_path + 'training.log'}")
    train_loader, test_loader = get_data(args)
    return experiment_log, args, log_path, train_loader, test_loader

def get_weight_update(model : DenseNN,
                      train_loader : DataLoader,
                      optimizer : torch.optim.SGD,
                      loss_fn : torch.nn.modules.loss._Loss | torch.nn.modules.loss._WeightedLoss,
                      args : dict) -> tuple[torch.Tensor, float]:
    """Get the weight update direction (full batch gradient descent) without updating the model parameters.

    Also returns the training loss for the epoch.
    return: weight update (torch.Tensor), epoch train loss (float)
    """
    model.train()
    model.zero_grad()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if not args["pre_transfer"]:
            data, target = data.to(args["device"]), target.to(args["device"])
        output = model(data)
        loss = loss_fn(output, target) / len(data)
        epoch_loss += loss * len(data)
        loss.backward()
    return (-optimizer.param_groups[0]['lr'] * cat_and_flatten([p.grad for p in model.parameters()])).clone(), epoch_loss.item()

def get_landscape_metrics(model : DenseNN,
                          normalised_weight_update : torch.Tensor,
                          test_loader : DataLoader,
                          optimizer : torch.optim.SGD,
                          loss_fn : torch.nn.modules.loss._Loss | torch.nn.modules.loss._WeightedLoss,
                          args : dict) -> tuple[torch.Tensor, float]:
    """Get test landscape metrics relating to slope and curvature
    
    Where v is the normalised_weight_update and <.,.> is the dot product:
    return: jacobian norm, <jacobian,v>, <v,Hv>, test loss
    """
    model.eval()
    model.zero_grad(set_to_none=False)
    test_loss = 0
    for data, target in test_loader:
        if not args["pre_transfer"]:
            data, target = data.to(args["device"]), target.to(args["device"])
        output = model(data)
        test_loss += loss_fn(output, target)
    # Compute gradient of loss function wrt model parameters (J)
    jacobian = torch.autograd.grad(test_loss / len(test_loader.dataset), model.get_params(), create_graph=True)
    j_norm = torch.linalg.norm(torch.cat([i.view(-1) for i in jacobian])).item()
    normalised_jacobian = nn.functional.normalize(torch.cat([i.view(-1) for i in jacobian]), dim=0)
    jv_product = torch.dot(normalised_jacobian, normalised_weight_update).item()
    hv_product = torch.autograd.grad(torch.cat([i.view(-1) for i in jacobian]), model.get_params(), grad_outputs=normalised_weight_update, retain_graph=True)
    vhv_product = torch.dot(normalised_weight_update, cat_and_flatten(hv_product)).item()
    return j_norm, jv_product, vhv_product, test_loss.item() / len(test_loader.dataset)

def update_model_parameters(model : DenseNN,
                            weight_update : torch.Tensor) -> None:
    with torch.no_grad():
        model_params = nn.utils.parameters_to_vector(model.parameters())
        model_params += weight_update
        nn.utils.vector_to_parameters(model_params, model.parameters())

def train_and_get_metrics(model : DenseNN,
        train_loader : DataLoader,
        test_loader : DataLoader,
        optimizer : torch.optim.SGD,
        loss_fn : torch.nn.modules.loss._Loss | torch.nn.modules.loss._WeightedLoss,
        args : dict,
        log_path : str,
        record_metrics : bool,
        end_condition : int) -> dict:
    """Run full batch gradient descent, taking landscape metrics at every epoch
    params:
        end_condition: (int) 0: End after record count reaches limit,
                             1: End after train loss reaches steady state,
                             2: End after train and test loss reach steady state
    return: dict with keys {"train_losses", 
                            "test_losses",
                            "train_steady_state_start",
                            "test_steady_state_start",
                            "jv_products",
                            "vhv_products",
                            "j_norms",
                            "weight_update_norms"}
    """
    train_losses = []
    test_losses = []
    weights = []
    weight_diffs = []
    save_model(model, f"{log_path + 'model_size_' + str(model.num_hidden_units)}_initial.pt")
    with trange(args["epochs"]) as pbar:
        for epoch in pbar:
            # Get weight update direction using the full batch of training data
            weight_update, epoch_loss = get_weight_update(model, train_loader, optimizer, loss_fn, args)
            train_losses.append(epoch_loss / len(train_loader.dataset))
            test_loss = test(model, test_loader, loss_fn, args)
            test_losses.append(test_loss)
            
            update_model_parameters(model, weight_update)
            
            if epoch % 10 == 0:
                weights.append(cat_and_flatten(model.get_params()))

            if epoch > 0:
                if epoch % 1000 == 0:
                    current_weights = cat_and_flatten(model.get_params())
                    for weight in weights:
                        diff = torch.sum(torch.abs(current_weights - weight)) / len(current_weights)
                        weight_diffs.append(diff.item())
                    weights = []

            if (epoch+1) % 1000 == 0:
                save_model(model, f"{log_path + 'model_size_' + str(model.num_hidden_units)}_{epoch}_epochs.pt")

            logging.info(f"epoch: {epoch}, train: {train_losses[-1]:.16f}, test: {test_losses[-1]:.16f}")
            pbar.set_postfix_str(f"train: {train_losses[-1]}, test: {test_losses[-1]}", refresh=True)
    save_model(model, f"{log_path + 'model_size_' + str(model.num_hidden_units)}_{epoch}_epochs.pt")
    output = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "weight_diffs": weight_diffs
    }
    return output

args = {
    "seed": 1,
    "data_seed": 2147483647,
    "train_size": 4000,
    "batch_size": 64,
    "test_batch_size": 1000,
    "pre_transfer": True,
    "epochs": 20000,
    "lr": 0.001,
    "momentum": 0.95,
    "gamma": 0.9,
    "loss_fn": "mse",
}

if __name__ == "__main__":
    for num_hidden_units in [100]:#, 40, 60, 80, 100, 120, 140]:
        experiment_log, args, log_path, train_loader, test_loader = setup_exp(args)
        #start_model = load_model(60, r"./log/weight_convergence_tests/2023-08-15_09;07;40/model_size_60_14999_epochs.pt").to(args["device"])
        model = DenseNN(num_hidden_units).to(args["device"])
        #model.set_params(extend_params(start_model.get_params(), num_hidden_units, args["device"]))
        optimizer = optim.SGD(model.parameters(), lr=args["lr"], momentum=args["momentum"])
        loss_fn = MSELoss(reduction="sum")
        # SET START/END AT STEADY STATE
        experiment_log[f"model_{model.num_hidden_units}"] = train_and_get_metrics(model, train_loader, test_loader, optimizer, loss_fn, args, log_path, True, 0)
        args.pop("device")
        save_json(experiment_log, f"{log_path + 'experiment_log'}.json")