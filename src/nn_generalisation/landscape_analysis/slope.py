from ..neural_net.neural_net import DenseNN
from ..experiment import Experiment

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def get_jacobian(model : DenseNN,
                 exp : Experiment) -> tuple[torch.Tensor]:
    params = model.get_params()
    loss = get_test_with_grad(model, exp)
    return torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

def get_test_with_grad(model : DenseNN,
                       exp : Experiment):
    model.eval()
    test_loss = 0
    for data, target in exp.test_loader:
        if not exp.args["pre_transfer"]:
            data, target = data.to(exp.args["device"]), target.to(exp.args["device"])
        output = model(data)
        test_loss += F.mse_loss(output, target, reduction="sum")
    return test_loss / len(exp.test_loader.dataset)