import tensorflow as tf

"""This script was developed based on:
https://www.tensorflow.org/guide/data_performance
https://www.tensorflow.org/datasets/performances
"""

class PreprocessImageDataset():
	
	def __init__(self, parameters):
		self.image_size = parameters["image_size"]
		self.batch_size: int = int(parameters["batch_size"])
		

	def normalize_img(self, image, label):
		"""
		Normalizes images: `uint8` -> `float32`.
		
		Keyword arguments:
			image (tf.Tensor): Multi-dimentional Tensor containing RGB values of images
			label (tf.Tensor): Labels for each image
		"""
		return tf.cast(tf.image.resize(image, (self.image_size[0], self.image_size[1])), tf.float32), label
	
	def optimize_train_set(self, train_dataset):
		"""
		Optimizes dataset loading time to avoid data starvation
		on the GPU by prefetching, batching and shuffling data
		ahead of time
		"""
		print("optimizing training")
		train_dataset = train_dataset.batch(self.batch_size).map(self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
		return train_dataset.shuffle(buffer_size=10).prefetch(tf.data.AUTOTUNE)
		
	def optimize_test_set(self, test_dataset):
		"""
		Optimizes dataset loading time to avoid data starvation
		on the GPU by batching, caching and prefetching data
		ahead of time.
		"""

		test_dataset = test_dataset.batch(self.batch_size).map(self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
		return test_dataset.prefetch(tf.data.AUTOTUNE)

	def optimize_validation_set(self, val_dataset):
		"""
		Optimizes dataset loading time to avoid data starvation
		on the GPU by batching, caching and prefetching data
		ahead of time.
		"""

		val_dataset = val_dataset.batch(self.batch_size).map(self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
		return val_dataset.prefetch(tf.data.AUTOTUNE)
