import os
import sys
import yaml
import mlflow
import argparse
from datetime import datetime
from Workflow import train, test, evaluate
from Datasets import TF_dataset_loader

# Get the path to root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the root to the system path
sys.path.append(project_root)

parser = argparse.ArgumentParser()
parser.add_argument("--yaml_path", type=str)
args = vars(parser.parse_args())

timestamp = datetime.now().strftime("%m:%d:%y-%H-%M-%S")  # %H:%M:%S")


def run_train(parameters, experiment_id, dataset):
    trainer = train.Train(parameters, experiment_id, dataset)
    trainer.train_routine(dataset)


def run_test(parameters, experiment_id, dataset):
    test.test_routine(parameters, experiment_id, dataset)


def evaluate_model(parameters, experiment_id, dataset):
    evaluate.eval_routine(parameters, experiment_id, dataset)


if __name__ == "__main__":
    with open(args["yaml_path"], "r") as file:
        # Parse the YAML data
        parameters = yaml.safe_load(file)

    
    experiment_id = mlflow.create_experiment(f"mlruns/{timestamp}")
    mlflow.start_run(experiment_id=experiment_id)

    dataset = TF_dataset_loader.TensorFlowDataset(parameters["train"])
    print(type(dataset))
    actions = {"train": run_train, "test": run_test, "eval": evaluate_model}

    for key in actions.keys():  # Runs the whole dict
        if parameters.get(key) is not None:  # Checks if input is in the yaml
            actions[key](
                parameters[key], experiment_id, dataset
            )  # Sends filtered parameters for each step

    mlflow.end_run()
