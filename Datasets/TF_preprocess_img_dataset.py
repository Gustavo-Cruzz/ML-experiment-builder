import tensorflow as tf
from typing import Tuple

"""This script was developed based on:
https://www.tensorflow.org/guide/data_performance
https://www.tensorflow.org/datasets/performances
"""


class PreprocessImageDataset:
    def __init__(self, parameters: dict) -> None:
        """
        Initializes PreprocessImageDataset with given parameters.

        Args:
            parameters (dict): Dictionary containing parameters for preprocessing.
                Expected keys: "image_size" (tuple), "batch_size" (int)
        """
        self.image_size: Tuple[int, int, int] = parameters["image_size"]
        self.batch_size: int = int(parameters["batch_size"])

    def normalize_img(
        self, image: tf.Tensor, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Normalizes images: `uint8` -> `float32`.

        Args:
            image (tf.Tensor): Multi-dimensional Tensor containing RGB values of images
            label (tf.Tensor): Labels for each image

        Returns:
            Tuple containing normalized image and label
        """
        return (
            tf.cast(
                tf.image.resize(image, (self.image_size[0], self.image_size[1])),
                tf.float32,
            ),
            label,
        )

    def optimize_train_set(self, train_dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Optimizes dataset loading time to avoid data starvation
        on the GPU by prefetching, batching and shuffling data
        ahead of time.

        Args:
            train_dataset (tf.data.Dataset): TensorFlow dataset containing training data

        Returns:
            train_dataset (tf.data.Dataset): Optimized TensorFlow dataset
        """
        train_dataset = train_dataset.batch(self.batch_size).map(
            self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE
        )
        return train_dataset.shuffle(buffer_size=10).prefetch(tf.data.AUTOTUNE)

    def optimize_test_set(self, test_dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Optimizes dataset loading time to avoid data starvation
        on the GPU by batching, caching and prefetching data
        ahead of time.

        Args:
            test_dataset (tf.data.Dataset): TensorFlow dataset containing training data

        Returns:
            test_dataset (tf.data.Dataset): Optimized TensorFlow dataset
        """
        test_dataset = test_dataset.batch(self.batch_size).map(
            self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE
        )
        return test_dataset.prefetch(tf.data.AUTOTUNE)

    def optimize_validation_set(self, val_dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Optimizes dataset loading time to avoid data starvation
        on the GPU by batching, caching and prefetching data
        ahead of time.

        Args:
            val_dataset (tf.data.Dataset): TensorFlow dataset containing training data

        Returns:
            val_dataset (tf.data.Dataset): Optimized TensorFlow dataset
        """
        val_dataset = val_dataset.batch(self.batch_size).map(
            self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE
        )
        return val_dataset.prefetch(tf.data.AUTOTUNE)
