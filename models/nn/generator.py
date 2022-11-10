from .baseInterfaces import AbstractDLModel
import tensorflow as tf
from tensorflow.keras import layers

class DCGANGenerator(AbstractDLModel):
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, input_shape: tuple=(100,), from_path=None):
        super().__init__(optimizer=optimizer, input_shape=input_shape, from_path=from_path)

    def _define_archtecture(self):
        return tf.keras.Sequential(layers=[
            layers.Dense(7*7*256, use_bias=False, input_shape=self.get_input_shape()),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Reshape((7, 7, 256)),
            
            # 7x7x128
            layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # 14x14x64
            layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), padding='same', use_bias=False, strides=(2, 2)),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            # 7x7x1
            layers.Conv2DTranspose(filters=1, kernel_size=(5, 5), padding='same', use_bias=False, strides=(2, 2), activation='tanh'),
        ])
