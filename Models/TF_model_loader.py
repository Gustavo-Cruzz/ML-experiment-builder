import os
import tensorflow as tf
from tensorflow.keras import Model
from typing import Tuple, Callable, Dict, Any, List, Union
from Models import TF_abstract_model
from sklearn.metrics import classification_report
from tensorflow.keras import applications as tf_app
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from pathlib import Path


class TensorFlowModel(TF_abstract_model.ABS_Model):

    def __init__(self, parameters: Dict[str, Any], data_shape: Union[Tuple[int, int, int], str]) -> None:
        """
        Initializes a TensorFlow model.

        Args:
            parameters (Dict[str, Any]): Dictionary containing model parameters.
            data_shape (Union[Tuple[int, int, int], str]): Shape of the input data.
        """
        if "device" in parameters:
            if parameters["device"] == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.epochs: int = parameters["epochs"]
        self.loss: str = parameters["loss_func"]
        self.output_shape: int = parameters["classes"]
        self.model_name: str = parameters["model_name"]
        self.batch_size: int = parameters["batch_size"]
        self.dataset_name: str = parameters["dataset_name"]
        self.activation: str = parameters["activation_func"]
        
        if isinstance(data_shape, str):
            self.input_shape = tuple(
                int(num) for num in data_shape.strip("[]").split(", ")
            )
        else:
            self.input_shape = data_shape
            
        self.create_model()

    def create_mobile_v2(self) -> Tuple[Model, Callable]:
        """
        Creates a MobileNetV2 model with the appropriate preprocessing layer.

        Returns:
            Tuple[Model, Callable]: Base model and preprocessing function.
        """
        base_model = tf_app.MobileNetV2(
            input_shape=self.input_shape, include_top=False, weights="imagenet"
        )
        preprocess_input = tf_app.mobilenet_v2.preprocess_input
        return (base_model, preprocess_input)

    def create_VGG16(self) -> Tuple[Model, Callable]:
        """
        Creates a VGG16 model with the appropriate preprocessing layer.

        Returns:
            Tuple[Model, Callable]: Base model and preprocessing function.
        """
        base_model = tf_app.VGG16(
            input_shape=self.input_shape, include_top=False, weights="imagenet"
        )
        preprocess_input = tf_app.vgg16.preprocess_input

        return (base_model, preprocess_input)

    def create_ResNet50(self) -> Tuple[Model, Callable]:
        """
        Creates a ResNet50 model with the appropriate preprocessing layer.

        Returns:
            Tuple[Model, Callable]: Base model and preprocessing function.
        """
        base_model = tf_app.ResNet50(
            input_shape=self.input_shape, include_top=False, weights="imagenet"
        )
        preprocess_input = tf_app.resnet50.preprocess_input

        return (base_model, preprocess_input)

    def create_model(self) -> None:
        """
        Loads a generic tensorflow model with a custom output layer.
        Any image model from https://www.tensorflow.org/api_docs/python/tf/keras/applications
        can be easily implemented.
        
        Raises:
            ValueError: If the model name is not supported.
        """

        model_dict: Dict[str, Callable[[], Tuple[Model, Callable]]] = {
            "mobile_netv2": self.create_mobile_v2,
            "VGG16": self.create_VGG16,
            "ResNet50": self.create_ResNet50,
        }
        print(f"Loading model: {self.model_name}")

        if self.model_name not in model_dict:
             raise ValueError(
                f"Invalid name for TensorFlowModel: {self.model_name}. Valid values: {list(model_dict.keys())}"
            )

        base_model, preprocessing_layer = model_dict[self.model_name]()

        for layer in base_model.layers:
            layer.trainable = False

        global_average_layer = GlobalAveragePooling2D()

        inputs = tf.keras.Input(shape=self.input_shape)
        x = preprocessing_layer(inputs)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = Dense(64)(x)
        x = Dropout(0.2)(x)

        outputs = Dense(self.output_shape, activation=self.activation)(x)
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(
            loss=self.loss,
            optimizer="Adam",
            metrics=["accuracy"],
        )

    def fit(self, dataset: Any) -> Any:
        """
        Trains the TensorFlow model on the given dataset.

        Args:
            dataset (Any): Object containing train, test, and val datasets.

        Returns:
            Any: Results of the training process (History object).
        """

        results = self.model.fit(
            dataset.train_dataset,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_data=dataset.val_dataset,
            shuffle=True,
        )

        return results

    def predict(self, dataset: Any) -> Any:
        """
        Generates predictions for the test dataset.

        Args:
            dataset (Any): Object containing train, test, and val datasets.
            
        Returns:
            Any: Model predictions.
        """
        return self.model.predict(dataset.test_dataset)

    def save_model(self, path: Union[str, Path], model_name: str) -> str:
        """
        Save the model in the given path.

        Args:
            path (Union[str, Path]): Path to save the model.
            model_name (str): Name of the model file.

        Returns:
            str: The full path used.
        """
        save_p = Path(path) / f"{model_name}.keras"
        self.model.save(save_p)
        return str(save_p)

    def get_metrics(self, dataset: Any, pred: Any) -> List[Dict[str, Any]]:
        """
        Calculates metrics from model predictions.

        Args:
            dataset (Any): Concrete Dataset class with x y data.
            pred (Any): List with model's predictions.
            
        Returns:
            List[Dict[str, Any]]: List with metrics dictionary.
        """
        labels = dataset.get_test_y()
        class_names = [str(i) for i in range(self.output_shape)]
        
        # Convert predictions to class indices if they are probabilities
        if len(pred.shape) > 1 and pred.shape[1] > 1:
            y_pred = pred.argmax(axis=1)
        else:
            y_pred = (pred > 0.5).astype(int)
            
        # Convert one-hot encoded labels to class indices if necessary
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            y_true = labels.argmax(axis=1)
        else:
            y_true = labels

        return [classification_report(y_true, y_pred, target_names=class_names, output_dict=True)]

    def load_model(self, path: Union[str, Path], model_name: str) -> None:
        """
        Load the model from the given path.

        Args:
            path (Union[str, Path]): Path to the model folder.
            model_name (str): Name of the model file.
        """
        self.model = tf.keras.models.load_model(Path(path) / model_name)
