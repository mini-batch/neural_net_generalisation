from .slope import get_jacobian, get_test_with_grad
from .utils import cat_and_flatten, get_number_of_parameters
from ..neural_net.neural_net import DenseNN
from ..experiment import Experiment 

import torch
from tqdm import tqdm

def get_hessian_avg(model : DenseNN, exp : Experiment) -> float:
    model = model.to(exp.args["device"])
    params = model.get_params()
    num_params = get_number_of_parameters(model.num_hidden_units)
    jacobian = get_jacobian(model, exp)
    jacobian = cat_and_flatten(jacobian)
    non_zero_indices = torch.nonzero(jacobian, as_tuple=True)[0].tolist()
    running_sum = 0
    for i in tqdm(non_zero_indices, desc="Calculate second derivative average"):
        curvature_i = cat_and_flatten(torch.autograd.grad(jacobian[i], params, retain_graph=True))
        running_sum += torch.sum(torch.abs(curvature_i))
    return running_sum.item() / (num_params * num_params)