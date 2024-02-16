import abc


class ABS_Model(abc.ABC):
    """Base abstract class for each architecture model that will be created.

    Args:
        abc (ABC): It's the abstract base class to create the model.
    """

    @abc.abstractmethod
    def fit(self, sample_x, sample_y):
        """This is an abstract method to fit the model

        Args:
            sample_x ({DataFrame, array}): The instance with the attributes
            that will train the model.
            sample_y (array): A array of shape with the labels for each
            instance
            that will train the model.
        """
        pass

    @abc.abstractmethod
    def predict(self, dataset_test):
        """This is an abstract method to predict the class labels.

        Args:
            dataset_test (array): The data matrix instance with the
            attributes that the model will get the predictions.
        """
        pass

    @abc.abstractmethod
    def get_metrics(self, dataset, pred):
        """This is an abstract method to get the metrics to be logged, pyaiutils API is suggested.

        Args:
            dataset (ABSDataset): The dataset object to get the true labels
            pred (Array): prediction array
        """
        pass

    @abc.abstractmethod
    def save_model(self, path, model_name):
        """Save the model and return the complete path to transform it in a artifact

        Args:
            path (string): path to save the model

                returns:
                save_p (string) the full path used
        """
        pass

    @abc.abstractmethod
    def load_model(self, path):
        """Load the model from the given path

        Args:
                path (string): path to the model folder

        """
        pass
