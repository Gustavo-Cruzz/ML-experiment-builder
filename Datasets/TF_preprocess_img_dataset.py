import tensorflow as tf
from typing import Tuple

"""This script was developed based on:
https://www.tensorflow.org/guide/data_performance
https://www.tensorflow.org/datasets/performances
"""


class PreprocessImageDataset:
    def __init__(self, parameters: dict, info=None) -> None:
        """
        Initializes PreprocessImageDataset with given parameters.

        Args:
            parameters (dict): Dictionary containing parameters for preprocessing.
                Expected keys: "image_size" (tuple), "batch_size" (int)
            info: Optional dataset information
        """
        self.image_size: Tuple[int, int, int] = parameters["image_size"]
        self.batch_size: int = int(parameters["batch_size"])
        self.Autotune = tf.data.experimental.AUTOTUNE
        self.info = info


    def convert_to_rgb(self, image: tf.Tensor, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Checks the number of channels and converts the image to 3-channel RGB 
        if it is currently 1-channel grayscale. This is necessary to use the imagenet
        weights in the models.
        """
        # Check static shape first to avoid graph construction errors
        if image.shape[-1] == 3:
            return image, label
            
        # Check the number of channels (the last dimension)
        image_shape = tf.shape(image)
        num_channels = image_shape[-1]
        
        # Define the condition: Check if the number of channels is 1
        is_grayscale = tf.equal(num_channels, 1)
        
        # If grayscale (True), use tf.image.grayscale_to_rgb (duplicates the channel)
        # If not grayscale (False), use the image as is
        image = tf.cond(
            is_grayscale, 
            lambda: tf.image.grayscale_to_rgb(image), 
            lambda: image
        )
            
        return image, label

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
        train_dataset = train_dataset.map(
            self.normalize_img, num_parallel_calls=self.Autotune
        )

        train_dataset = train_dataset.map(
            self.convert_to_rgb, num_parallel_calls=self.Autotune 
        )
        
        train_dataset.shuffle(self.info.splits["train"].num_examples)
        return train_dataset.batch(self.batch_size).prefetch(self.Autotune)

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
        test_dataset = test_dataset.map(
            self.normalize_img, num_parallel_calls=self.Autotune
        )

        test_dataset = test_dataset.map(
            self.convert_to_rgb, num_parallel_calls=self.Autotune 
        )

        return test_dataset.batch(self.batch_size).prefetch(self.Autotune)

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
        val_dataset = val_dataset.map(
            self.normalize_img, num_parallel_calls=self.Autotune
        )

        val_dataset = val_dataset.map(
            self.convert_to_rgb, num_parallel_calls=self.Autotune
        )

        return val_dataset.batch(self.batch_size).prefetch(self.Autotune)
