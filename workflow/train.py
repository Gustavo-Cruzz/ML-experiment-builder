# Third party imports
import os
import mlflow
import tempfile
import numpy as np
import pyaiutils as pyai
import matplotlib.pyplot as plt
from Models.TF_model_loader import TensorFlowModel
from typing import Dict, Any, List
from Datasets import TF_dataset_loader


class Train():
    def __init__(self, params: Dict,  experiment_id: str) -> None:
        self.params = params
        self.save_path = f"..{self.params['save_path']}/{experiment_id}/train/"
        print(self.params)
        
        self.train_routine()
        
    def log_data(
        self, metrics: List, pred: List[Any], temp_dir: str
    ) -> None:
        """
        Logs data including predictions, parameters, and metrics to the specified save path.

        Args:
            metrics (List): List of metrics.
            pred (List[Any]): List of predictions.
            temp_dir (str): Temporary directory path.
        """
        prediction_file_path = os.path.join(self.save_path, "prediction.txt")
        with open(prediction_file_path, "w") as file:
            file.write("\n".join(map(str, pred)))

        mlflow.log_param("parameters", self.params)
        mlflow.log_param("save_path", self.save_path)
        mlflow.log_artifacts(self.save_path)
        mlflow.log_dict(metrics[0].to_df())  # Log all metrics


    def plot_training_history(self, results: Any, temp_dir: str) -> None:
        """
        Plots the training history and saves the plots to the temporary directory.

        Args:
            results (Any): Results of the training process.
            temp_dir (str): Temporary directory path.
        """
        print("type_results", type(results))
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


    def plot_training_graphics(
        self, 
        y_test: np.ndarray,
        pred: np.ndarray,
        temp_dir: str,
        model: TensorFlowModel,
    ) -> None:
        """
        Plots various training graphics and saves them to the temporary directory.

        Args:
            y_test (np.ndarray): Test data.
            pred (np.ndarray): Predicted data.
            temp_dir (str): Temporary directory path.
            model (TensorflowModel): Trained model.
        """
        n_classes = [i for i in range(model.output_shape)]

        pyai.plot_confusion_matrix(
            y_test, pred, save_path=temp_dir, title="Confusion Matrix"
        )
        pyai.plot_auc_roc_multi_class(
            y_test, pred, save_path=temp_dir, class_names=n_classes
        )
        pyai.plot_prc_auc_multiclass(
            y_test, pred, save_path=temp_dir, class_names=n_classes
        )


    def train_routine(self) -> None:
        """
        Executes the training routine.

        Args:
            experiment_id (str): Experiment ID.
        """

        model = TensorFlowModel(self.params)
        print(f"\n Training model, {model}")

        dataset = TF_dataset_loader.TensorFlowDataset(self.params)

        # Fit, predict and log metrics and model
        results = model.fit(dataset)

        pred = model.predict(dataset)

        metrics = model.get_metrics(dataset, pred)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.plot_training_history(results, temp_dir)
            self.plot_training_graphics(dataset.get_test_y(), pred, temp_dir, model)

            model.save_model(temp_dir, "Base_model")
            self.log_data(temp_dir, metrics, pred)
