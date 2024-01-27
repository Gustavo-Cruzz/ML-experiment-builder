# Third party imports
import tensorflow_datasets as tfds
import tensorflow as tf
from TF_preprocess_img_dataset import PreprocessImageDataset

# Application specific imports.
from utils.dataset.abs_dataset import ABSDataset

class TensorFlowDataset(ABSDataset):
	
	def __init__(self, parameters):
		self.dataset_type: str = parameters["dataset_type"]
		self.dataset_name: str = parameters["dataset_name"]
		self.target: str = parameters["target"]
		self.batch_size: int = int(parameters["batch_size"])

		self.__create_dataset()
		
		if self.dataset_type.upper() == "IMAGE":
			results = PreprocessImageDataset(parameters, self.train, self.val_dataset, self.test)
			self.train = results['train']
			self.val_dataset = results['val_dataset']
			self.test = results['test']

	def __create_dataset(self):
		#TODO checar funcionalidade do shuffle e proportion
		try:
			(ds_train, ds_test), info = tfds.load(
				self.dataset_name, split=['train', 'test'],
				as_supervised=True, with_info=True, shuffle_files=True
			)
			proportion = int(len(ds_train) * .2)

			# Shuffle now to avoid getting only one class on split
			
			ds_train = ds_train.shuffle(10)
			self.val_dataset = ds_train.take(proportion)
			self.train_dataset = ds_train.skip(proportion)
			self.test_dataset = ds_test

		except tfds.core.registered.DatasetNotFoundError:
			raise Exception("Dataset not found in tensorflow_datasets; Ensure that the dataset has 'train' or 'test' split")
	
	def get_train_data(self):
		"""Returns the train dataset"""
		return self.train_dataset

	def get_test_data(self):
		"""Returns the test dataset"""
		return self.test_dataset

	@tf.autograph.experimental.do_not_convert
	def get_train_x(self):
		"""Returns the train features"""
		train_data = self.get_train_data()
		return list(train_data.map(lambda image, _: image))

	@tf.autograph.experimental.do_not_convert
	def get_train_y(self):
		"""Returns unprocessed and unoptimized train targets"""
		train_data = list(self.train_data.map(lambda _, label: label))
		return list([item for sublist in train_data for item in sublist])

	@tf.autograph.experimental.do_not_convert
	def get_test_x(self):
		"""Returns the test features"""
		test_data = self.get_test_data()
		return list(test_data.map(lambda image, _: image))

	@tf.autograph.experimental.do_not_convert
	def get_test_y(self):
		"""Returns unprocessed and unoptimized test targets"""
		test_data = list(self.test_dataset.map(lambda _, label: label))
		return list([item for sublist in test_data for item in sublist])