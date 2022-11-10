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

