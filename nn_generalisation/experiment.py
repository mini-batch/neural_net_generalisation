from . import neural_net
from .data import data_utils, logging_utils

import logging
from datetime import datetime
import os
from tqdm import tqdm, trange
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import MSELoss


class Experiment:
    """
    Top level class to contain environment for experimentation
    """
    def __init__(self, args):
        set_seeds(args["seed"])
        self.args = set_device(args)
        self.train_loader, self.test_loader = data_utils.get_data(self.args)

        # Initialise models
        self.models : list[neural_net.DenseNN] = []
        for num in self.args["num_hidden_units"]:
            self.models.append(neural_net.DenseNN(num).to(self.args["device"]))

        if args["param_reuse"]:
            self.last_model_params = None

        self.datetime = get_datetime_str()
        self.experiment_log = {"args": args}

    def run(self):
        self.log_path = setup_log_path(self.datetime)
        logging_utils.setup_log(f"{self.log_path + 'training.log'}")
        for model in tqdm(self.models, desc="Experiment Progress"):
            logging.info(f"{model}")
            if self.args["param_reuse"]:
                run_with_param_reuse(model, self)
            else:
                run_without_param_reuse(model, self)
        self.args.pop("device")
        return logging_utils.save_json(self.experiment_log, f"{self.log_path + 'experiment_log'}.json")


def train_model(model : neural_net.DenseNN,
                train_loader : DataLoader,
                test_loader : DataLoader,
                args : dict,
                log_path : str) -> tuple[list[float], list[float]]:
    # Save model with initialised weights for testing equality of output
    logging_utils.save_model(model, f"{log_path + 'model_size_' + str(model.num_hidden_units)}_initial.pt")

    train_losses = []
    test_losses = []
    optimizer = optim.SGD(model.parameters(), lr=args["lr"], momentum=args["momentum"])
    if args["loss_fn"] == "mse":
        loss_fn = MSELoss(reduction="sum")
    else:
        raise Experiment("Specified loss function was not valid")
    for epoch in trange(args["epochs"], desc=f"Model with {model.num_hidden_units} hidden units training progress"):
        train_loss = neural_net.train(model, train_loader, optimizer, loss_fn, args)
        test_loss = neural_net.test(model, test_loader, loss_fn, args)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        logging.info(f"epoch: {epoch}, train: {train_loss:.16f}, test: {test_loss:.16f}")
    
    # Save model with final weights for testing equality of output
    logging_utils.save_model(model, f"{log_path + 'model_size_' + str(model.num_hidden_units)}_final.pt")
    return train_losses, test_losses


def run_with_param_reuse(model : neural_net.DenseNN,
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
        exp.last_model_params = neural_net.extend_params(exp.last_model_params, model.num_hidden_units, exp.args["device"])
        model.set_params(exp.last_model_params)
    exp.experiment_log[f"model_{model.num_hidden_units}"] = train_model(model, exp.train_loader, exp.test_loader, exp.args, exp.log_path)
    exp.last_model_params = model.get_params()


def run_without_param_reuse(model : neural_net.DenseNN,
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
