from src.nn_generalisation.experiment import Experiment
from experiment_args import experiment_args
from src.nn_generalisation.landscape_analysis.slope import get_jacobian, plot_grad
from src.nn_generalisation.logging.models import load_model

import torch


def main():
    exp = Experiment(experiment_args)
    # exp.run()
    model = load_model(10, "./log/2023-07-11_11;36;18/model_size_10_final.pt")
    jacobian = get_jacobian(model, exp)
    flat_jacob = torch.cat([torch.flatten(i) for i in jacobian])
    plot_grad(flat_jacob.numpy(force=True), 1000)

if __name__ == "__main__":
    main()