import torch
import torch.nn as nn

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
