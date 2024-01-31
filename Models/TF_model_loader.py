import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import applications as tf_app
from keras.utils import to_categorical
import pyaiutils
import os
from Models import TF_abstract_model 

class TensorFlowModel(TF_abstract_model.ABS_Model):

	def __init__(self, parameters):	
		if "device" in parameters:
			if parameters["device"] == "cpu":
				os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
		
		self.model_name = parameters["model_name"]
		self.dataset_name = parameters["dataset_name"]
		self.input_shape = parameters["image_size"]
		self.output_shape = parameters["classes"]
		self.batch_size = parameters["batch_size"]
		self.epochs = parameters["epochs"]

		if type(self.input_shape) is str:
			self.input_shape =  [int(num) for num in self.input_shape.strip("[]").split(", ")]
		self.create_model() 


	def create_mobile_v2(self):
		base_model = tf_app.MobileNetV2(input_shape=self.input_shape,
										include_top=False, weights='imagenet')
		preprocess_input = tf_app.mobilenet_v2.preprocess_input

		return (base_model, preprocess_input)

	def create_VGG16(self):
		base_model = tf_app.VGG16(input_shape=self.input_shape,
										include_top=False, weights='imagenet')
		preprocess_input = tf_app.vgg16.preprocess_input

		return (base_model, preprocess_input)

	def create_ResNet50(self):
		base_model = tf_app.ResNet50(input_shape=self.input_shape,
										include_top=False, weights='imagenet')
		preprocess_input =  tf_app.resnet50.preprocess_input

		return (base_model, preprocess_input)

	def create_model(self):
		""" Loads a generic backbone model with custom output layers
			models from https://www.tensorflow.org/api_docs/python/tf/keras/applications 
		"""

		model_dict = {"mobile_netv2": self.create_mobile_v2,
									"VGG16": self.create_VGG16,
									"ResNet50": self.create_ResNet50}
			
		base_model, preprocessing_layer = model_dict.get(self.model_name)()

		if base_model is None:
			raise Exception(f"""Invalid name for TensorFlowModel
									 			 Valid values: {model_dict.keys()}""")
		
		for layer in base_model.layers:
			layer.trainable = False	

		global_average_layer = GlobalAveragePooling2D()

		inputs = tf.keras.Input(shape=self.input_shape)
		x = preprocessing_layer(inputs)
		x = base_model(x,training=False)
		x = global_average_layer(x)
		x = Dense(64)(x)
		x = Dropout(0.2)(x)
		
		outputs = Dense(self.output_shape, activation="softmax")(x)
		
		self.model = tf.keras.Model(inputs, outputs)
		self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
							 optimizer="Adam", metrics = ["accuracy"])

	def fit(self, dataset):
		"""
		Keyword arguments:
			dataset (tf_dataset): Object containing train,test and val datasets
		"""
		
		results =  self.model.fit(
					dataset.train_dataset,
					batch_size=self.batch_size,
					epochs=self.epochs,
					verbose=1,
					validation_data=dataset.val_dataset,
					shuffle=True
				)
		return results

	def predict(self, dataset):
		"""
		Keyword arguments:
			dataset (tf_dataset): Object containing train,test and val datasets
		"""
		return self.model.predict(dataset.test_dataset)
	
	def save_model(self, path, model_name):
		"""Save the model in the given path

		Args:
			path (string): path to save the model
		
		returns:
		save_p (string) the full path used
		"""
		save_p = os.path.join(path, model_name + '.keras')
		self.model.save(save_p)
		return save_p
	
	def get_metrics(self, dataset, pred):
		"""
		Calculates metrics from model predictions

		Keyword arguments:
			dataset (Dataset): Concrete Dataset class with x y data
			pred (list): List with model's predictions
		"""

		labels = dataset.get_test_y()
		one_hot_labels = to_categorical(labels, self.output_shape)
		class_names = [i for (i,j) in enumerate(range(0, self.output_shape))]
		return [pyaiutils.get_metrics(one_hot_labels, pred, class_names = class_names)]
		
	def load_model(self,path):
		"""Load the model from the given path

		Args:
			path (string): path to the model folder

		"""
		self.model = tf.keras.models.load_model(os.path.join(path,"Base_model.keras"))
