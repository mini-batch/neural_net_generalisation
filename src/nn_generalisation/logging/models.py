from ..neural_net.neural_net import DenseNN

import torch

def save_model(model : DenseNN, path : str) -> None:
    torch.save(model.state_dict(), path)

def load_model(num_hidden_units, path : str) -> DenseNN:
    model = DenseNN(num_hidden_units)
    model.load_state_dict(torch.load(path))
    return model