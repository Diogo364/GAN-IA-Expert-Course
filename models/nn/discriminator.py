from .baseInterfaces import AbstractSelfArchitectureDLModel, CNNAbstractDLModel
import tensorflow as tf
from tensorflow.keras import layers

class DCGANDiscriminator(AbstractSelfArchitectureDLModel):
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

class Pix2PixDiscriminator(CNNAbstractDLModel):
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, input_shape=(256, 256, 3), from_path=None):
        super().__init__(optimizer, input_shape, from_path)
    
    def _define_archtecture(self):
        initializer = tf.random_normal_initializer(0, 0.02)
        original_input = layers.Input(shape=self.get_input_shape(), name='original_img')
        transformed_input = layers.Input(shape=self.get_input_shape(), name='transformed_img')
        
        x = layers.Concatenate()([original_input, transformed_input]) # 256, 256, 6
        down1 = self._create_conv_block(64, 4, False)(x)      # 128, 128, 64 
        down2 = self._create_conv_block(128, 4, False)(down1) # 64, 64, 128 
        down3 = self._create_conv_block(256, 4, False)(down2) # 32, 32, 256 
        zero_pad1 = layers.ZeroPadding2D()(down3)             # 34, 34, 256 
        down4 = self._create_conv_block(512, 4, True, padding='valid', strides=1, use_bias=True)(zero_pad1) # 31, 31, 512
        zero_pad2 = layers.ZeroPadding2D()(down4) # 33, 33, 512
        last = layers.Conv2D(1, 4, kernel_initializer=initializer, strides=1)(zero_pad2) # 30, 30, 1

        return tf.keras.Model(inputs=[original_input, transformed_input], outputs=last)