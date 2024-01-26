import argparse
import mlflow
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--yaml_path", type=str)
args = vars(parser.parse_args())

with open(args['yaml_path'], 'r') as file:
	# Parse the YAML data
	parameters = yaml.safe_load(file)

def _run(entrypoint, parameters={}, source_version=None, use_cache=True):
	"""Launching new run for an entrypoint"""

	print(
		"Launching new run for entrypoint=%s and parameters=%s"
		% (entrypoint, parameters)
	)
	submitted_run = mlflow.run(".", entrypoint, parameters={"yaml_path": args["yaml_path"]})
	return submitted_run


if __name__ == "__main__":

	valid_entries = ["train", "test", "eval"]
	actions = parameters.keys()

	actions = [key for key in actions if key in valid_entries]
	
	for ind, action in enumerate(actions):
		
		_run(action, parameters[action])   