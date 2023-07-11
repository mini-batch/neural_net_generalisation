import torch

def extend_params(params : list[torch.Tensor], new_model_size : int) -> list[torch.Tensor]:
    """
    Extend parameters to shape of larger model, new parameters are set to 0.
    """
    hidden_unit_diff = new_model_size - params[0].size(dim=0)
    params[0] = torch.cat([params[0], torch.zeros((hidden_unit_diff, 784))], dim=0)
    assert params[0].shape == (new_model_size, 784), f"Expected {(new_model_size, 784)} but got {params[0].shape}"
    params[1] = torch.cat([params[1], torch.zeros((hidden_unit_diff))], dim=0)
    assert params[1].shape[0] == (new_model_size), f"Expected {new_model_size} but got {params[1].shape}"
    params[2] = torch.cat([params[2], torch.zeros(10, hidden_unit_diff)], dim=1)
    assert params[2].shape == (10, new_model_size), f"Expected {(10, new_model_size)} but got {params[2].shape}"
    return params