# Third party imports
from typing import List, Dict, Any, Tuple
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical

from Datasets import Abstract_dataset
from Datasets import TF_preprocess_img_dataset


class TensorFlowDataset(Abstract_dataset.ABS_Dataset):

    def __init__(self, parameters: Dict[str, Any]) -> None:
        """
        Initializes the TensorFlowDataset.

        Args:
            parameters (Dict[str, Any]): Dictionary containing dataset parameters.
        """
        self.dataset_name: str = parameters["dataset_name"]
        self.output_shape: int = parameters["classes"]
        self.batch_size: int = parameters["batch_size"]

        self.__create_dataset()

        preprocess_images = TF_preprocess_img_dataset.PreprocessImageDataset(parameters, self.info)

        self.train_dataset = preprocess_images.optimize_train_set(self.train_dataset)
        self.val_dataset = preprocess_images.optimize_validation_set(self.val_dataset)
        self.test_dataset = preprocess_images.optimize_test_set(self.test_dataset)

    def __create_dataset(self) -> None:
        """
        Loads the dataset from tensorflow_datasets and splits it into train, validation, and test sets.
        
        Raises:
            Exception: If the dataset is not found in tensorflow_datasets.
        """
        try:
            (ds_train, ds_test), self.info = tfds.load(
                self.dataset_name,
                split=["train", "test"],
                as_supervised=True,
                with_info=True,
                shuffle_files=True,
            )
            
            train_size = self.info.splits["train"].num_examples
            proportion = int(train_size * 0.2)

            self.val_dataset = ds_train.take(proportion) # Validation set with 20% of train data
            self.train_dataset = ds_train.skip(proportion)

            self.test_dataset = ds_test

        except tfds.core.registered.DatasetNotFoundError:
            raise Exception(
                """Dataset not found in tensorflow_datasets; 
                Ensure that the dataset has 'train' or 'test' split"""
            )

    def get_train_data(self) -> tf.data.Dataset:
        """
        Returns:
            tf.data.Dataset: The training dataset.
        """
        return self.train_dataset

    def get_test_data(self) -> tf.data.Dataset:
        """        
        Returns:
            tf.data.Dataset: The test dataset.
        """
        return self.test_dataset

    @tf.autograph.experimental.do_not_convert
    def get_train_x(self) -> List[Any]:
        """
        Returns:
            List[Any]: List of training features.
        """
        train_data = self.get_train_data()
        return list(train_data.map(lambda image, _: image))

    @tf.autograph.experimental.do_not_convert
    def get_train_y(self) -> Any:
        """
        Returns:
            Any: One-hot encoded training labels.
        """
        train_data = list(self.train_dataset.map(lambda _, label: label))
        return to_categorical(
            [item for sublist in train_data for item in sublist], self.output_shape
        )

    @tf.autograph.experimental.do_not_convert
    def get_test_x(self) -> List[Any]:
        """
        Returns:
            List[Any]: List of test features.
        """
        test_data = self.get_test_data()
        return list(test_data.map(lambda image, _: image))

    @tf.autograph.experimental.do_not_convert
    def get_test_y(self) -> Any:
        """
        Returns:
            Any: One-hot encoded test labels.
        """
        test_data = list(self.test_dataset.map(lambda _, label: label))
        return to_categorical(
            [item for sublist in test_data for item in sublist], self.output_shape
        )

    def get_data_shape(self) -> Tuple[int, ...]:
        """
        Returns:
            Tuple[int, ...]: Shape of the input data (excluding batch dimension).
        """
        for image, _ in self.train_dataset.take(1):
            return image.shape[1:]  # Exclude batch dimension