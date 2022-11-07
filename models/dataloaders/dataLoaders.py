from abc import ABC, abstractmethod
import tensorflow as tf

class DataLoaderInterface(ABC):
    @abstractmethod
    def load_data(self):
        pass


class MNISTDataLoader(DataLoaderInterface):
    def load_data(self):
        return tf.keras.datasets.mnist.load_data()
