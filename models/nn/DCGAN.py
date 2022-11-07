from typing import Callable
from .baseInterfaces import AbstractDLModel
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class DCGANGenerator(AbstractDLModel):
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, input_shape: tuple=(100,)):
        super().__init__(optimizer=optimizer, input_shape=input_shape)

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
    

class DCGANDiscriminator(AbstractDLModel):
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, input_shape=(28, 28, 1)):
        super().__init__(optimizer, input_shape)
    
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
    

class GANPipeline():
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
    
    @tf.function
    def train(self, images):
        
        
        random_noise = tf.random.normal(np.ravel([[images.shape[0]], self._generator.get_input_shape()]))
        
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self._generator.predict(random_noise, training=True)
            
            expected_output = self._discriminator.predict(images, training=True)
            fake_output = self._discriminator.predict(generated_images, training=True)

            gen_loss = self._apply_generator_loss(fake_output)
            
            disc_loss = self._apply_discriminator_loss(expected_output, fake_output)

        self._generator.update_weights(gen_tape, gen_loss)
        self._discriminator.update_weights(disc_tape, disc_loss)