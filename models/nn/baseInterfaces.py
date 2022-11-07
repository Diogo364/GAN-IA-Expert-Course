from abc import ABC, abstractmethod
import tensorflow as tf

class ModelInterface(ABC):
    
    @abstractmethod
    def _define_archtecture(self):
        pass
    
    @abstractmethod
    def predict(self, x, **kwargs):
        pass


class AbstractDLModel(ModelInterface,ABC):
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, input_shape:tuple):
        self._optimizer = optimizer
        self._input_shape = input_shape
        self.model = self._define_archtecture()
    
    def predict(self, x, **kwargs):
        return self.model(x, **kwargs)
    
    def update_weights(self, tape, loss_value):
        generator_gradient = tape.gradient(loss_value, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(generator_gradient, self.model.trainable_variables))
        
    def get_input_shape(self):
        return self._input_shape