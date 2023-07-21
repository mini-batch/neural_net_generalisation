from ..neural_net import DenseNN

import logging
import json
import torch

def save_json(obj, path : str) -> None:
    with open(path, "w") as outfile:
        outfile.write(json.dumps(obj, indent=4))

def load_json(path : str) -> dict:
    with open(path, "r") as infile:
        return json.load(infile)
    
def save_model(model : DenseNN, path : str) -> None:
    torch.save(model.state_dict(), path)

def load_model(num_hidden_units, path : str) -> DenseNN:
    model = DenseNN(num_hidden_units)
    model.load_state_dict(torch.load(path))
    return model

def setup_log(path : str) -> None:
    logging.basicConfig(filename=path,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG,
                            force=True)