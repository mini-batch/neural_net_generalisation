from .neural_net.neural_net import DenseNN
from .data.load_data import get_data
from .utils import set_seeds, set_device, get_datetime_str, setup_log_path, run_with_param_reuse, run_without_param_reuse
from .logging.metrics import save_json
from .logging.utils import setup_log

import logging
from tqdm import tqdm

class Experiment:
    """
    Top level class to contain environment for experimentation
    """
    def __init__(self, args):
        set_seeds(args["seed"])
        self.args = set_device(args)
        self.train_loader, self.test_loader = get_data(self.args)

        # Initialise models
        self.models : list[DenseNN] = []
        for num in self.args["num_hidden_units"]:
            self.models.append(DenseNN(num))

        if args["param_reuse"]:
            self.last_model_params = None

        self.datetime = get_datetime_str()
        self.experiment_log = {"args": args}

    def run(self):
        self.log_path = setup_log_path(self.datetime)
        setup_log(f"{self.log_path + 'training.log'}")
        for model in tqdm(self.models, desc="Experiment Progress"):
            logging.info(f"{model}")
            if self.args["param_reuse"]:
                run_with_param_reuse(model, self)
            else:
                run_without_param_reuse(model, self)
        self.args.pop("device")
        return save_json(self.experiment_log, f"{self.log_path + 'experiment_log'}.json")
