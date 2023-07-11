from .neural_net.neural_net import DenseNN
from .neural_net.train import train
from .neural_net.test import test
from .neural_net.utils import extend_params
from .logging.models import save_model

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from datetime import datetime
import os
import logging
from tqdm import tqdm, trange

def train_model(model : DenseNN,
                train_loader : DataLoader,
                test_loader : DataLoader,
                args : dict,
                log_path : str) -> tuple[list[float], list[float]]:
    # Save model with initialised weights for testing equality of output
    save_model(model, f"{log_path + 'model_size_' + str(model.num_hidden_units)}_initial.pt")

    train_losses = []
    test_losses = []
    optimizer = optim.SGD(model.parameters(), lr=args["lr"], momentum=args["momentum"])
    for epoch in trange(args["epochs"], desc=f"Model with {model.num_hidden_units} hidden units training progress"):
        train_loss = train(model, train_loader, optimizer, args)
        test_loss = test(model, test_loader, args)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        logging.info(f"epoch: {epoch}, train: {train_loss:.16f}, test: {test_loss:.16f}")
    
    # Save model with final weights for testing equality of output
    save_model(model, f"{log_path + 'model_size_' + str(model.num_hidden_units)}_final.pt")
    return train_losses, test_losses


def run_with_param_reuse(model : DenseNN,
                         exp : 'Experiment') -> None:
    """
    Given a model and experiment, initialise the model with a transformation
    of the previous model and run training.
    params:
        model: (DenseNN)
        exp: (Experiment)
    """
    if exp.last_model_params != None:
        # Initialise model with transformed weights of previous model
        exp.last_model_params = extend_params(exp.last_model_params, model.num_hidden_units)
        model.set_params(exp.last_model_params)
    exp.experiment_log[f"model_{model.num_hidden_units}"] = train_model(model, exp.train_loader, exp.test_loader, exp.args, exp.log_path)
    exp.last_model_params = model.get_params()


def run_without_param_reuse(model : DenseNN,
                            exp : 'Experiment') -> None:
    """
    Given a model and experiment, initialise the model randomly and run training.
    params:
        model: (DenseNN)
        exp: (Experiment)
    """
    exp.experiment_log[f"model_{model.num_hidden_units}"] = train_model(model, exp.train_loader, exp.test_loader, exp.args, exp.log_path)    


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


def get_datetime_str() -> str:
    date_now = str(datetime.now().date())
    time_now = str(datetime.now().time())[:-7].replace(":", ";")
    return f"{date_now}_{time_now}"


def setup_log_path(datetime : str) -> str:
    os.makedirs(f"./log/{datetime}/", exist_ok = True)
    return f"./log/{datetime}/"