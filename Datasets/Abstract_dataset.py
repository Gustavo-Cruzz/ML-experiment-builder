import abc
from typing import Any


class ABSDataset(abc.ABC):

    @abc.abstractmethod
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def get_train_data(self) -> Any:
        pass

    @abc.abstractmethod
    def get_test_data(self) -> Any:
        pass

    @abc.abstractmethod
    def get_train_x(self) -> Any:
        pass

    @abc.abstractmethod
    def get_train_y(self) -> Any:
        pass

    @abc.abstractmethod
    def get_test_x(self) -> Any:
        pass

    @abc.abstractmethod
    def get_test_y(self) -> Any:
        pass
