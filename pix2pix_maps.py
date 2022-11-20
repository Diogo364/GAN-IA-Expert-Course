import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils import normalize, random_crop, random_flip, load_env
import matplotlib.pyplot as plt
from models.dataloaders import MapsURLDataLoader
from models.preprocessor import GenericPreprocessor
from models.nn.generator import DCGANGenerator
from models.nn.discriminator import DCGANDiscriminator
from models.nn.gan_pipeline import WGAN_GPPipeline
from utils.custom_losses import WassersteinDiscriminatorLoss, WassersteinGeneratorLoss

load_env()

def get_maps_data():
    generic_train_preprocessor = GenericPreprocessor(preprocessor_list=[random_crop, random_flip, normalize])
    generic_test_preprocessor = GenericPreprocessor(preprocessor_list=[normalize])
    (train, test) = MapsURLDataLoader(filename=os.environ['PIX2PIX_MAPS_FILE'], url=f"{os.environ['PIX2PIX_DATASETS_URL']}/{os.environ['PIX2PIX_MAPS_FILE']}").load_data()
    train = train.map(generic_train_preprocessor)
    test = test.map(generic_test_preprocessor)
    return train, test

if __name__ == '__main__':
    OUTPATH = osp.join(os.environ['OUTPATH'], 'Fashion-MNIST_WGAN-GP')
    WGAN_ASSETS = osp.join(os.environ['ASSETS'], 'Fashion-MNIST_WGAN-GP')

    os.makedirs(OUTPATH, exist_ok=True)
    os.makedirs(WGAN_ASSETS, exist_ok=True)

    test_size = 16
    batch_size = 1
    buffer_size = 1096
    random_noise_size = 100
    epochs = 100
    save_artefact_frequency = 5 # each n epochs

    train, test = get_maps_data()
    
    train = train.shuffle(buffer_size).batch(batch_size)
    test = test.batch(batch_size)

# TODO
    # test_images = tf.random.normal([test_size, random_noise_size])

    # generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    # discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

    # generator_loss = WassersteinGeneratorLoss()
    # discriminator_loss = WassersteinDiscriminatorLoss()

    # generator = DCGANGenerator(optimizer=generator_optimizer, input_shape=(random_noise_size,))
    # discriminator = DCGANDiscriminator(optimizer=discriminator_optimizer)

    # pipeline = WGAN_GPPipeline(generator, 
    #                         discriminator, 
    #                         generator_loss, 
    #                         discriminator_loss)

    # for epoch in tqdm(range(epochs), total=epochs, desc='Training Epoch'):
        
    #     for image_batch in tqdm(dataset, total=len(dataset), desc='Batches', leave=False):
    #         pipeline.train(image_batch)
    #         break
        
    #     if (epoch == 0) or ((epoch+1) % save_artefact_frequency == 0):
    #         test_output = generator.predict(test_images)
    #         fig, axes = plt.subplots(4, 4, figsize=(20,20), facecolor='white')
    #         for out_fig, ax in zip(test_output, axes.ravel()):
    #             ax.imshow(out_fig, cmap='gray')
    #             ax.axis('off')
    #         fig.savefig(osp.join(OUTPATH, f'epoch_{epoch+1}_out_grid.png'))
    #         plt.close(fig)

    # pipeline._generator.save_model(osp.join(WGAN_ASSETS, 'saved_model.hdf5'))