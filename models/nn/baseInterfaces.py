from abc import ABC, abstractmethod
import tensorflow as tf

class ModelInterface(ABC):
    
    @abstractmethod
    def _define_archtecture(self):
        pass
    
    @abstractmethod
    def predict(self, x, **kwargs):
        pass

    @abstractmethod
    def save_model(self, path):
        pass
    
    @abstractmethod
    def load_model(self, path):
        pass


class AbstractDLModel(ModelInterface,ABC):
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, input_shape:tuple, from_path: str=None):
        self._optimizer = optimizer
        self._input_shape = input_shape
        self.model = self.load_model(from_path) if from_path else self._define_archtecture()            
    
    def predict(self, x, **kwargs):
        return self.model(x, **kwargs)
    
    def update_weights(self, tape, loss_value):
        generator_gradient = tape.gradient(loss_value, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(generator_gradient, self.model.trainable_variables))
        
    def get_input_shape(self):
        return self._input_shape

    def save_model(self, path):
        self.model.save(path)
    
    def load_model(self, path):
        return tf.keras.models.load_model(path)


class GANPipelineInterface(ABC):
    @abstractmethod
    def _apply_discriminator_loss(self, expected_output, fake_output):
        pass
    
    @abstractmethod
    def _apply_generator_loss(self, fake_output):
        pass    
    
    @abstractmethod
    def train(self, images):
        pass

class AbstractGANPipeline(GANPipelineInterface, ABC):
    def __init__(self, 
                generator: AbstractDLModel, 
                discriminator: AbstractDLModel, 
                generator_loss: tf.keras.losses.Loss, 
                discriminator_loss=tf.keras.losses.Loss):
        self._generator = generator
        self._generator_loss = generator_loss
        
        self._discriminator = discriminator
        self._discriminator_loss = discriminator_loss

    def _apply_discriminator_loss(self, expected_output, fake_output):
        real_loss = self._discriminator_loss(tf.ones_like(expected_output), expected_output)
        fake_loss = self._discriminator_loss(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
    
    def _apply_generator_loss(self, fake_output):
        return self._generator_loss(tf.ones_like(fake_output), fake_output)