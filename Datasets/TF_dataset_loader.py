# Third party imports
from typing import List
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.utils import to_categorical

from Datasets import Abstract_dataset
from Datasets import TF_preprocess_img_dataset


class TensorFlowDataset(Abstract_dataset.ABSDataset):

    def __init__(self, parameters):
        self.dataset_type: str = parameters["dataset_type"]
        self.dataset_name: str = parameters["dataset_name"]
        self.output_shape: int = parameters["classes"]
        preprocess_images = TF_preprocess_img_dataset.PreprocessImageDataset(parameters)

        self.__create_dataset()

        if self.dataset_type.upper() == "IMAGE":
            self.train_dataset = preprocess_images.optimize_train_set(
                self.train_dataset
            )
            self.val_dataset = preprocess_images.optimize_validation_set(
                self.val_dataset
            )
            self.test_dataset = preprocess_images.optimize_test_set(self.test_dataset)

    def __create_dataset(self):
        # TODO checar funcionalidade do shuffle e proportion
        try:
            (ds_train, ds_test), info = tfds.load(
                self.dataset_name,
                split=["train", "test"],
                as_supervised=True,
                with_info=True,
                shuffle_files=True,
            )
            proportion = int(len(ds_train) * 0.2)

            # Shuffle now to avoid getting only one class on split

            ds_train = ds_train.shuffle(10)
            self.val_dataset = ds_train.take(proportion)
            self.train_dataset = ds_train.skip(proportion)
            self.test_dataset = ds_test

        except tfds.core.registered.DatasetNotFoundError:
            raise Exception(
                "Dataset not found in tensorflow_datasets; Ensure that the dataset has 'train' or 'test' split"
            )

    def get_train_data(self) -> tf.data.Dataset:
        """Returns the train dataset"""
        return self.train_dataset

    def get_test_data(self) -> tf.data.Dataset:
        """Returns the test dataset"""
        return self.test_dataset

    @tf.autograph.experimental.do_not_convert
    def get_train_x(self) -> List:
        """Returns the train features"""
        train_data = self.get_train_data()
        return list(train_data.map(lambda image, _: image))

    @tf.autograph.experimental.do_not_convert
    def get_train_y(self) -> List:
        """Returns unprocessed and unoptimized train targets"""
        train_data = list(self.train_data.map(lambda _, label: label))
        return to_categorical(
            [item for sublist in train_data for item in sublist], self.output_shape
        )

    @tf.autograph.experimental.do_not_convert
    def get_test_x(self) -> List:
        """Returns the test features"""
        test_data = self.get_test_data()
        return list(test_data.map(lambda image, _: image))

    @tf.autograph.experimental.do_not_convert
    def get_test_y(self) -> List:
        """Returns unprocessed and unoptimized test targets"""
        test_data = list(self.test_dataset.map(lambda _, label: label))
        return to_categorical(
            [item for sublist in test_data for item in sublist], self.output_shape
        )
