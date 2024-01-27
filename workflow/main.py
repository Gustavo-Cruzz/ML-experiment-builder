import argparse
import mlflow
import yaml
import train, test, evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--yaml_path", type=str)
args = vars(parser.parse_args())

def _run(entrypoint, parameters={}, source_version=None, use_cache=True):
	"""Launching new run for an entrypoint"""

	print(f"Launching new run for {entrypoint} and parameters={parameters}")
	return mlflow.run(".", entrypoint, parameters={"yaml_path": args["yaml_path"]})


def run_train(parameters):
	train.train_function(parameters)

def run_test():
	pass

def evaluate_model():
	pass

if __name__ == "__main__":
	with open(args['yaml_path'], 'r') as file:
		# Parse the YAML data
		parameters = yaml.safe_load(file)

	mlflow.start_run()
	valid_entries = ["train", "test", "eval"]

	actions = {"train": run_train,
						"test": run_test,
						"eval": evaluate_model}
	
	for key in parameters.keys():
		
		actions[key](parameters)  

	mlflow.end_run()