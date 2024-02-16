# Third party imports
import os
import mlflow
import tempfile
import numpy as np
import pyaiutils as pyai
import matplotlib.pyplot as plt
from Models import TF_model_loader
from Datasets import TF_dataset_loader


def log_data(save_path, metrics, params, pred, temp_dir):
    prediction_file_path = os.path.join(save_path, "prediction.txt")
    with open(prediction_file_path, "w") as file:
        file.write("\n".join(map(str, pred)))

    mlflow.log_param("parameters", params)
    mlflow.log_param("save_path", save_path)
    mlflow.log_artifacts(temp_dir)
    mlflow.log_dict(metrics[0].to_df())  # Log all metrics


def plot_training_history(results, temp_dir):
    for key, value in results.history.items():

        if key[:3] == "val":  # Skip validation indices
            continue

        final_value = np.round(value[-1], 2)
        plt.figure(figsize=(10, 8))
        plt.title(f"Model's {key} - {final_value}")
        plt.xlabel("Epochs")
        plt.ylabel("Values")
        plt.plot(value, label="key")
        plt.plot(results.history[f"val_{key}"], label=f"val_{key}")
        plt.legend()
        plt.savefig(temp_dir)
        plt.close()


def plot_training_graphics(y_test, pred, temp_dir, model):
    n_classes = [i for i in range(model.output_shape)]

    print(f"Temporary directory created: {temp_dir}")
    pyai.plot_confusion_matrix(
        y_test, pred, save_path=temp_dir, title="Confusion Matrix"
    )
    pyai.plot_auc_roc_multi_class(
        y_test, pred, save_path=temp_dir, class_names=n_classes
    )
    pyai.plot_prc_auc_multiclass(
        y_test, pred, save_path=temp_dir, class_names=n_classes
    )


def train_routine(params, experiment_id):
    print(params)

    save_path = f"..{params['save_path']}/{experiment_id}/train/"

    model = TF_model_loader.TensorFlowModel(params)
    print(f"\n Training model, {model}")

    # dataset = load_data(parameters["dataset_file_path"], parameters) #TODO
    dataset = TF_dataset_loader.TensorFlowDataset(params)

    # Fit, predict and log metrics and model
    results = model.fit(dataset)

    pred = model.predict(dataset)

    metrics = model.get_metrics(dataset, pred)

    with tempfile.TemporaryDirectory() as temp_dir:
        print("created dir", temp_dir)
        print(f"save_path: {save_path}")
        plot_training_history(results, temp_dir)

        plot_training_graphics(dataset.get_test_y(), pred, temp_dir, model)

        model.save_model(temp_dir, "Base_model")
        log_data(temp_dir, metrics, params, pred)
