import yaml
import mlflow
import argparse
import train, test, evaluate
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--yaml_path", type=str)
args = vars(parser.parse_args())

timestamp = datetime.now().strftime("%m:%d:%y-%H-%M-%S")  # %H:%M:%S")


def run_train(parameters, experiment_id):
    train.train_routine(parameters, experiment_id)


def run_test():
    test.test_routine(parameters, experiment_id)


def evaluate_model():
    evaluate.eval_routine(parameters, experiment_id)


if __name__ == "__main__":
    with open(args["yaml_path"], "r") as file:
        # Parse the YAML data
        parameters = yaml.safe_load(file)

    experiment_id = mlflow.create_experiment(f"mlruns/{timestamp}")
    mlflow.start_run(experiment_id=experiment_id)

    actions = {"train": run_train, "test": run_test, "eval": evaluate_model}

    for key in actions.keys():  # Runs the whole dict
        if parameters.get(key) is not None:  # Checks if input is in the yaml
            actions[key](
                parameters[key], experiment_id
            )  # Sends filtered parameters for each step

    mlflow.end_run()
