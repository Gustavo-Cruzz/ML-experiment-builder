import os
import sys
import yaml
import mlflow
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Get the path to root directory
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from Workflow import train, test, evaluate
from Datasets import TF_dataset_loader

parser = argparse.ArgumentParser()
parser.add_argument("--yaml_path", type=str, required=True)
args = vars(parser.parse_args())

timestamp = datetime.now().strftime("%m:%d:%y-%H-%M-%S")


def run_train(parameters: Dict[str, Any], experiment_id: str, dataset: Any) -> None:
    """
    Runs the training workflow.

    Args:
        parameters (Dict[str, Any]): Training parameters.
        experiment_id (str): MLflow experiment ID.
        dataset (Any): Dataset object.
    """
    trainer = train.Train(parameters, experiment_id, dataset)

def run_test(parameters: Dict[str, Any], experiment_id: str, dataset: Any) -> None:
    """
    Runs the testing workflow.

    Args:
        parameters (Dict[str, Any]): Testing parameters.
        experiment_id (str): MLflow experiment ID.
        dataset (Any): Dataset object.
    """
    test.test_routine(parameters, experiment_id, dataset)


def evaluate_model(parameters: Dict[str, Any], experiment_id: str, dataset: Any) -> None:
    """
    Runs the evaluation workflow.

    Args:
        parameters (Dict[str, Any]): Evaluation parameters.
        experiment_id (str): MLflow experiment ID.
        dataset (Any): Dataset object.
    """
    evaluate.eval_routine(parameters, experiment_id, dataset)


if __name__ == "__main__":
    yaml_path = Path(args["yaml_path"])
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration file not found at: {yaml_path}")

    with open(yaml_path, "r") as file:
        # Parse the YAML data
        parameters = yaml.safe_load(file)

    
    experiment_id = mlflow.create_experiment(f"mlruns/{timestamp}")
    mlflow.start_run(experiment_id=experiment_id)

    if "train" not in parameters:
         raise ValueError("Configuration file must contain a 'train' section to initialize the dataset.")

    dataset = TF_dataset_loader.TensorFlowDataset(parameters["train"])
    
    actions = {"train": run_train, "test": run_test, "eval": evaluate_model}

    for key in actions.keys():  # Runs the whole dict
        if parameters.get(key) is not None:  # Checks if input is in the yaml
            actions[key](
                parameters[key], experiment_id, dataset
            )  # Sends filtered parameters for each step

    mlflow.end_run()
