from .baseInterfaces import AbstractDLModel, UNetAbstractDLModel
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

class Pix2PixGenerator(UNetAbstractDLModel):
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, input_shape: tuple=(256,256,3), from_path=None):
        super().__init__(optimizer=optimizer, input_shape=input_shape, from_path=from_path)

    def _define_archtecture(self):
        inputs = layers.Input(shape=(256,256,3))
        downsampling = [
            self._create_conv_block(64, 4, batch_norm=False), # 128, 128, 64
            self._create_conv_block(128, 4, batch_norm=True), # 64, 64, 128
            self._create_conv_block(256, 4, batch_norm=True), # 32, 32, 256
            self._create_conv_block(512, 4, batch_norm=True), # 16, 16, 512
            self._create_conv_block(512, 4, batch_norm=True), # 8, 8, 512
            self._create_conv_block(512, 4, batch_norm=True), # 4, 4, 512
            self._create_conv_block(512, 4, batch_norm=True), # 2, 2, 512
            self._create_conv_block(512, 4, batch_norm=True), # 1, 1, 512
        ]

        upsampling = [
            self._create_deconv_block(512, 4, dropout=True),  # 2, 2, 512
            self._create_deconv_block(512, 4, dropout=True),  # 4, 4, 512
            self._create_deconv_block(512, 4, dropout=True),  # 8, 8, 512
            self._create_deconv_block(512, 4, dropout=False), # 16, 16, 512
            self._create_deconv_block(256, 4, dropout=False), # 32, 32, 256
            self._create_deconv_block(128, 4, dropout=False), # 64, 64, 128
            self._create_deconv_block( 64, 4, dropout=False), # 128, 128, 64
        ]
        out_layer = layers.Conv2DTranspose( 3, 4, strides=2, padding='same', 
                                            kernel_initializer=tf.random_normal_initializer(0, 0.02), 
                                            activation='tanh') # 256, 256, 3

        x = inputs
        skips = []
        for down in downsampling:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])

        for up, skip in zip(upsampling, skips):
            x = up(x)
            x = layers.Concatenate()([x, skip])
        x = out_layer(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x)