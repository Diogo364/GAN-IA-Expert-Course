
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import os.path as osp
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from models.dataloaders import MNISTDataLoader
from models.nn import DCGANGenerator, DCGANDiscriminator, GANPipeline
from utils import normalize
import matplotlib.pyplot as plt


OUTPATH = osp.join('etc', 'DCGAN')

test_size = 16
batch_size = 256
buffer_size = batch_size * 3
random_noise_size = 100
epochs = 100

os.makedirs(OUTPATH, exist_ok=True)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
discriminator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

mnist_loader = MNISTDataLoader()
(x_train, y_train), (_, _) = mnist_loader.load_data()

x_train = np.expand_dims(normalize(x_train), axis=-1)


dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size=buffer_size).batch(batch_size=batch_size)

test_images = tf.random.normal([test_size, random_noise_size])

generator = DCGANGenerator(optimizer=generator_optimizer, input_shape=(random_noise_size,))
discriminator = DCGANDiscriminator(optimizer=discriminator_optimizer)

pipeline = GANPipeline( generator, 
                        discriminator, 
                        generator_loss, 
                        discriminator_loss)

for epoch in tqdm(range(epochs), total=epochs, desc='Training Epoch'):
    
    for image_batch in tqdm(dataset, total=len(dataset), desc='Batches', leave=False):
        pipeline.train(image_batch)
    
    test_output = generator.predict(test_images)
    fig, axes = plt.subplots(4, 4, figsize=(20,20), facecolor='white')
    for out_fig, ax in zip(test_output, axes.ravel()):
        ax.imshow(out_fig, cmap='gray')
        ax.axis('off')
    fig.savefig(osp.join(OUTPATH, f'epoch_{epoch+1}_out_grid.png'))
    plt.close(fig)