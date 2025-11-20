# ML Experiment Builder

## Overview
This project is a Machine Learning experiment builder designed to streamline the process of training, testing, and evaluating TensorFlow models. It leverages **MLflow** for experiment tracking and **TensorFlow** for model building and training. The system allows users to define experiments via YAML configuration files, making it easy to reproduce and iterate on different model architectures and hyperparameters.

## Installation

To set up the environment for this project, it is recommended to use Conda. You can create the environment using the provided configuration file:

```bash
conda env create -f InputFiles/Envs/TF_env.yaml
conda activate <env_name>
```

## Usage

There are two main ways to run experiments: using MLflow or running the Python script directly.

### Option 1: Using MLflow (Recommended)
You can run the project as an MLflow project. This ensures that the environment and entry points are handled correctly.

```bash
mlflow run . -P yaml_path=InputFiles/MobileNetV2.yaml
```

### Option 2: Direct Python Execution
You can also run the `main.py` script directly, providing the path to your configuration YAML file.

```bash
python main.py --yaml_path InputFiles/MobileNetV2.yaml
```

## Configuration

The project uses YAML files to configure experiments. These files are located in the `InputFiles` directory. Below is an explanation of the configuration parameters based on `InputFiles/MobileNetV2.yaml`:

```yaml
train:
  model_name: mobile_netv2          # Name of the model architecture to use
  activation_func: softmax          # Activation function for the output layer
  loss_func: sparse_categorical_crossentropy # Loss function for training
  classes: 10                       # Number of output classes
  save_path: mlruns                 # Directory to save MLflow runs
  batch_size: 200                   # Batch size for training
  epochs: 1                         # Number of training epochs
  dataset_name: cifar10             # Name of the dataset to load
  image_size: [224, 224, 3]         # Input image dimensions [height, width, channels]
  device: cpu                       # Device to run on (e.g., 'cpu', 'gpu')
  skip: False                       # Whether to skip the training step
```

### Supported Actions
The `main.py` script checks for the following keys in the YAML file to determine which steps to execute:
- `train`: Runs the training routine.
- `test`: Runs the testing routine.
- `eval`: Runs the evaluation routine.

## Project Structure

- **`main.py`**: The main entry point of the application. It parses the YAML configuration and triggers the appropriate workflow steps (train, test, eval).
- **`MLproject`**: Defines the MLflow project, including the environment and entry points.
- **`Workflow/`**: Contains the logic for the different stages of the experiment:
  - `train.py`: Handles model training, logging metrics to MLflow, and plotting training history.
  - `test.py`: Handles model testing.
  - `evaluate.py`: Handles model evaluation.
- **`Models/`**: Contains TensorFlow model definitions and loaders (`TF_model_loader.py`, `TF_abstract_model.py`).
- **`Datasets/`**: Contains data loading logic (`TF_dataset_loader.py`).
- **`InputFiles/`**: Stores configuration YAML files (e.g., `MobileNetV2.yaml`, `ResNet50.yaml`) and environment definitions.
- **`Utils/`**: Utility scripts, including graph generation (`Create_Graphs.py`).
- **`mlruns/`**: Directory where MLflow stores experiment runs and artifacts.