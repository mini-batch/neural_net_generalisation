from .slope import get_jacobian, get_test_with_grad
from .utils import cat_and_flatten, get_number_of_parameters
from ..neural_net.neural_net import DenseNN
from ..experiment import Experiment 

import torch
from tqdm import trange

def get_hessian_avg(model : DenseNN, exp : Experiment) -> float:
    model = model.to(exp.args["device"])
    params = model.get_params()
    num_params = get_number_of_parameters(model.num_hidden_units)
    jacobian = get_jacobian(model, exp)
    jacobian = cat_and_flatten(jacobian)
    avg = 0
    for i in trange(num_params, desc="Calculate second derivative"):
        curvature_i = cat_and_flatten(torch.autograd.grad(jacobian[i], params, retain_graph=True))
        avg += torch.mean(torch.abs(curvature_i)).item()
    return avg / (num_params * num_params)