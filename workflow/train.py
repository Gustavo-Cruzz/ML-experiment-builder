# Third party imports
import mlflow
import os
from datetime import datetime
from Models import TF_model_loader
from Datasets import TF_dataset_loader

def log_data(save_path, metrics, params, pred):
  prediction_file_path = os.path.join(save_path, "prediction.txt")
  with open(prediction_file_path, "w") as file:
        file.write("\n".join(map(str, pred)))

  mlflow.log_param("parameters", params)
  mlflow.log_param("save_path", save_path) 

  for df in metrics: # Merics come as a dataframe
    data = df.to_dict(orient='records') 
    for i in range(len(data)):
      if data[i]['Classes'] == 'Média': #Here we save the averages of each one in a dict
            data[i].pop('Classes', None)
            data[i] = {k + 'mean': v for k, v in data[i].items()}
      mlflow.log_metrics(data[i]) #Log metrics individually
    
 
def train_routine(params, experiment_id):
  print(params)
  # exit()
  save_path = f"..{params['save_path']}/{experiment_id}/train/"

  model = TF_model_loader.TensorFlowModel(params)
  print(f"\n Training model, {model}")

  # dataset = load_data(parameters["dataset_file_path"], parameters) #TODO
  dataset = TF_dataset_loader.TensorFlowDataset(params)  

  # Fit, predict and log metrics and model
  model.fit(dataset)

  pred = model.predict(dataset)
  metrics = model.get_metrics(dataset, pred)

  if not os.path.isdir(save_path):
      os.makedirs(save_path)
      
  model.save_model(save_path, "Base_model")

  log_data(save_path, metrics, params, pred)