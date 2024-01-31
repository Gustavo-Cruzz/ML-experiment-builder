# Third party imports
import os
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from Models import TF_model_loader
from Datasets import TF_dataset_loader

def log_data(save_path, metrics, params, pred):
  prediction_file_path = os.path.join(save_path, "prediction.txt")
  with open(prediction_file_path, "w") as file:
        file.write("\n".join(map(str, pred)))

  mlflow.log_param("parameters", params)
  mlflow.log_param("save_path", save_path) 

  mlflow.log_dict(metrics[0].to_df()) #Log all metrics

def generate_train_graphics(results, save_path):  
  for key, value in results.history.items():

    if key[:3] == "val": #Skip validation indices
      continue
  
    final_value = np.round(value[-1], 2)
    plt.figure(figsize=(10,8))
    plt.title(f"Model's {key} - {final_value}")
    plt.xlabel("Epochs")
    plt.ylabel("Values")
    plt.plot(value, label="key")
    plt.plot(results.history[f"val_{key}"], label=f"val_{key}")
    plt.legend()
    fig = plt.plot()
    mlflow.log_figure(fig, f"{key}.png")
    plt.close()

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

  generate_train_graphics(results, save_path)

  model.save_model(save_path, "Base_model")
  log_data(save_path, metrics, params, pred)
