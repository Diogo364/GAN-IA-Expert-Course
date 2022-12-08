from typing import Tuple
from utils.custom_losses import GradientPenaltyLossInterface
from .baseInterfaces import AbstractGANPipeline, AbstractDLModel
import numpy as np
import tensorflow as tf

class DCGANPipeline(AbstractGANPipeline):
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


class WGAN_GPPipeline(AbstractGANPipeline):
    def __init__(self, 
                generator: AbstractDLModel, 
                discriminator: AbstractDLModel, 
                generator_loss: tf.keras.losses.Loss, 
                discriminator_loss=GradientPenaltyLossInterface,
                discriminator_trains_per_epoch=3):
        self._discriminator_trains_per_epoch = discriminator_trains_per_epoch
        super().__init__(generator, discriminator, generator_loss, discriminator_loss)

    @tf.function
    def train(self, images):
        random_noise = tf.random.normal(np.ravel([[images.shape[0]], self._generator.get_input_shape()]))
        
        for _ in range(self._discriminator_trains_per_epoch):
            with tf.GradientTape() as disc_tape:
                generated_images = self._generator.predict(random_noise, training=True)
                
                expected_output = self._discriminator.predict(images, training=True)
                fake_output = self._discriminator.predict(generated_images, training=True)
                epsilon = tf.random.normal((images.shape[0], 1, 1, 1), dtype='float64')
                gradient_penalty = self._gradient_penalty(images, generated_images, epsilon)
                self._discriminator_loss.gradient_penalty = gradient_penalty
                disc_loss = self._discriminator_loss(expected_output, fake_output)
            self._discriminator.update_weights(disc_tape, disc_loss)
        
        with tf.GradientTape() as gen_tape:
            generated_images = self._generator.predict(random_noise, training=True)
            fake_output = self._discriminator.predict(generated_images, training=True)
            gen_loss = self._generator_loss(fake_output, fake_output)
        self._generator.update_weights(gen_tape, gen_loss)
        

    @tf.function
    def _gradient_penalty(self, real, fake, epsilon):
        interpolated_images = (real * epsilon) + (tf.cast(fake, dtype='float64') * (1 - epsilon))
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            scores = self._discriminator.predict(interpolated_images)

        gradient = tape.gradient(scores, interpolated_images)[0]
        gradient_norm = tf.norm(gradient)
        return tf.math.reduce_mean((gradient_norm -1)**2)

class Pix2PixPipeline(AbstractGANPipeline):
    def __init__(self, 
                generator: AbstractDLModel, 
                discriminator: AbstractDLModel, 
                generator_loss: tf.keras.losses.Loss, 
                p2p_loss: tf.keras.losses.Loss, 
                discriminator_loss: GradientPenaltyLossInterface,
                lambda_: float):
        super().__init__(generator, discriminator, generator_loss, discriminator_loss)
        self._p2p_loss = p2p_loss
        self._lambda = lambda_

    @tf.function
    def train(self, images: Tuple):
        raw_images, processed_images = images
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self._generator.predict(raw_images, training=True)
            expected_output = self._discriminator.predict([raw_images, processed_images], training=True)
            fake_output = self._discriminator.predict([raw_images, generated_images], training=True)
            
            gen_loss = self._apply_generator_loss(expected_output)
            pix2pix_loss = self._apply_pix2pix_loss(generated_images, processed_images)
            total_gen_loss = self._join_generator_losses(gen_loss, pix2pix_loss)
            
            disc_loss = self._discriminator_loss(expected_output, fake_output)
        
        self._generator.update_weights(gen_tape, total_gen_loss)
        self._discriminator.update_weights(disc_tape, disc_loss)

    def _apply_discriminator_loss(self, expected_output, fake_output):
        return super()._apply_discriminator_loss(expected_output, fake_output)

    def _apply_generator_loss(self, fake_output):
        return super()._apply_generator_loss(fake_output)

    def _apply_pix2pix_loss(self, generator_output, target):
        return self._p2p_loss(target, generator_output)

    def _join_generator_losses(self, gen_loss, pix2pix_loss):
        return gen_loss + (self._lambda * pix2pix_loss)