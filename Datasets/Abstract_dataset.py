import abc
from typing import Any


class ABSDataset(abc.ABC):
    """
    Abstract base class defining methods for dataset operations.
    """

    @abc.abstractmethod
    def __init__(self) -> None:
        """
        Constructor method for initializing the dataset object.
        """
        pass

    @abc.abstractmethod
    def get_train_data(self) -> Any:
        """
        Abstract method to retrieve training data from the dataset.

        Returns:
            Any: Training data
        """
        pass

    @abc.abstractmethod
    def get_test_data(self) -> Any:
        """
        Abstract method to retrieve test data from the dataset.

        Returns:
            Any: Test data
        """
        pass

    @abc.abstractmethod
    def get_train_x(self) -> Any:
        """
        Abstract method to retrieve training input features from the dataset.

        Returns:
            Any: Training input features
        """
        pass

    @abc.abstractmethod
    def get_train_y(self) -> Any:
        """
        Abstract method to retrieve training labels from the dataset.

        Returns:
            Any: Training labels
        """
        pass

    @abc.abstractmethod
    def get_test_x(self) -> Any:
        """
        Abstract method to retrieve test input features from the dataset.

        Returns:
            Any: Test input features
        """
        pass

    @abc.abstractmethod
    def get_test_y(self) -> Any:
        """
        Abstract method to retrieve test labels from the dataset.

        Returns:
            Any: Test labels
        """
        pass
