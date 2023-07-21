import torch
import numpy as np

def cat_and_flatten(input : tuple[torch.Tensor]):
	return torch.cat([torch.flatten(i) for i in input])

def get_mean_abs(input : torch.Tensor) -> float:
	input = input.numpy(force=True)
	return np.mean(np.abs(input))

def get_number_of_parameters(num_hidden_units):
	"""
	Number of parameters in the network for MNIST (dimension = 784, number of classes = 10),
	get the number of parameters in a model with num_hidden_units number of hidden units.
	params:
		num_hidden_units: (int)
	return (int)
	"""
	return (784 + 1) * num_hidden_units + (num_hidden_units + 1) * 10
