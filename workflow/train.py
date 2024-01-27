# Third party imports
import mlflow
import os
import sys
import yaml
from datetime import datetime
from ..models import TF_model_loader
import time

# Application specific imports.
# from misc import load_model, load_data, get_model_size
# from graphics import plot_graphics

timestamp = datetime.now().strftime("%m/%d/%y-%H-%M-%S") #%H:%M:%S")

def log_data(save_path, metrics, params, pred):
  prediction_file_path = os.path.join(save_path, "prediction.txt")
  with open(prediction_file_path, "w") as file:
        file.write("\n".join(map(str, pred)))

  mlflow.log_params("parameters", params)
  mlflow.log_artifacts("save_path", save_path) 
  mlflow.log_metric("metrics", metrics) 
 

def train_routine(params):
  save_path = f"{params["save_path"]}/{timestamp}"

  model = TF_model_loader.TensorflowModel(params)
  print(f"\n Training model, {model}")

  # dataset = load_data(parameters["dataset_file_path"], parameters) #TODO
  dataset = None  

  # Fit, predict and log metrics and model
  model.fit(dataset)

  metrics = model.get_metrics(dataset, pred)

  if not os.path.isdir(save_path):
      os.makedirs(save_path)
      
  model.save_model(save_path, "Base_model")

  pred = model.predict(dataset)

  log_data(save_path, metrics, params, pred)

