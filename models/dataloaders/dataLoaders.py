from abc import ABC, abstractmethod
from utils import remove_extension_names
import os.path as osp
import tensorflow as tf

class DataLoaderInterface(ABC):
    @abstractmethod
    def load_data(self):
        pass


class MNISTDataLoader(DataLoaderInterface):
    def load_data(self):
        return tf.keras.datasets.mnist.load_data()

class FashionMNISTDataLoader(DataLoaderInterface):
    def load_data(self):
        return tf.keras.datasets.fashion_mnist.load_data()
