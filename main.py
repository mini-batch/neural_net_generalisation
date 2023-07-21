from nn_generalisation.experiment import Experiment
from experiment_args import experiment_args


def main():
    exp = Experiment(experiment_args)
    exp.run()
    
    
if __name__ == "__main__":
    main()