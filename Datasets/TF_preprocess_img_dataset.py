import tensorflow as tf


class PreprocessImageDataset():
	
	def __init__(self, parameters, train, val, test):
		self.image_size = parameters["image_size"]
		
		self.train_dataset = train
		self.test_dataset = test
		self.val_dataset = val

		self.optimize_train_set()
		self.optimize_validation_set()
		self.optimize_test_set()

		return {'train': self.train_dataset, 'val_dataset': self.val_dataset, 'test': self.test_dataset}

	def normalize_img(self, image, label):
		"""
		Normalizes images: `uint8` -> `float32`.
		
		Keyword arguments:
			image (tf.Tensor): Multi-dimentional Tensor containing RGB values of images
			label (tf.Tensor): Labels for each image
		"""
		return tf.cast(tf.image.resize(image, (self.image_size[0], self.image_size[1])), tf.float32), label
	
	def optimize_train_set(self):
		"""
		Optimizes dataset loading time to avoid data starvation
		on the GPU by prefetching, batching and shuffling data
		ahead of time
		"""
		print("optimizing training")
		self.train_dataset = self.train_dataset.batch(self.batch_size).map(self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
		self.train_dataset = self.train_dataset.shuffle(buffer_size=10).prefetch(tf.data.AUTOTUNE)
		
	def optimize_test_set(self):
		"""
		Optimizes dataset loading time to avoid data starvation
		on the GPU by batching, caching and prefetching data
		ahead of time.
		"""

		self.test_dataset = self.test_dataset.batch(self.batch_size).map(self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
		self.test_dataset = self.test_dataset.prefetch(tf.data.AUTOTUNE)

	def optimize_validation_set(self):
		"""
		Optimizes dataset loading time to avoid data starvation
		on the GPU by batching, caching and prefetching data
		ahead of time.
		"""

		self.val_dataset = self.val_dataset.batch(self.batch_size).map(self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
		self.val_dataset = self.val_dataset.prefetch(tf.data.AUTOTUNE)
