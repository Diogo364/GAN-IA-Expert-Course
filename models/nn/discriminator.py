from .baseInterfaces import AbstractDLModel
import tensorflow as tf
from tensorflow.keras import layers

class DCGANDiscriminator(AbstractDLModel):
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, input_shape=(28, 28, 1), from_path=None):
        super().__init__(optimizer, input_shape, from_path)
    
    def _define_archtecture(self):
        return tf.keras.Sequential(layers=[
            # 14x14x64
            layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=self.get_input_shape()),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            
            # 7x7x128
            layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            layers.Flatten(),

            layers.Dense(1)
        ])
