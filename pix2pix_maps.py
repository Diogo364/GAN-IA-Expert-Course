import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils import normalize, random_crop, random_flip, load_env, tf_normalize
import matplotlib.pyplot as plt
from models.dataloaders import MapsURLDataLoader
from models.preprocessor import GenericPreprocessor
from models.nn.generator import Pix2PixGenerator
from models.nn.discriminator import Pix2PixDiscriminator
from models.nn.gan_pipeline import Pix2PixPipeline
from utils.custom_losses import MeanAbsoluteErrorLoss

load_env()

def get_maps_data():
    generic_train_preprocessor = GenericPreprocessor(preprocessor_list=[random_crop, random_flip, normalize])
    generic_test_preprocessor = GenericPreprocessor(preprocessor_list=[normalize])
    (train, test) = MapsURLDataLoader(filename=os.environ['PIX2PIX_MAPS_FILE'], url=f"{os.environ['PIX2PIX_DATASETS_URL']}/{os.environ['PIX2PIX_MAPS_FILE']}").load_data()
    train = train.map(generic_train_preprocessor)
    test = test.map(generic_test_preprocessor)
    return train, test

if __name__ == '__main__':
    OUTPATH = osp.join(os.environ['OUTPATH'], 'Maps_PIX2PIX')
    PIX2PIX_ASSETS = osp.join(os.environ['ASSETS'], 'Maps_PIX2PIX')

    os.makedirs(OUTPATH, exist_ok=True)
    os.makedirs(PIX2PIX_ASSETS, exist_ok=True)

    test_size = 16
    batch_size = 1
    buffer_size = 1096
    random_noise_size = (256,256,3)
    epochs = 40000
    save_artefact_frequency = 5 # each n epochs
    learning_rate = 0.0002
    beta1, beta2 = 0.5, 0.999
    lambda_ = 100

    train, test = get_maps_data()
    
    train = train.shuffle(buffer_size).batch(batch_size)
    test = test.batch(batch_size)

    test_images = test.take(test_size)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)

    generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    pix2pix_loss = MeanAbsoluteErrorLoss()
    discriminator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator = Pix2PixGenerator(optimizer=generator_optimizer, input_shape=random_noise_size)
    discriminator = Pix2PixDiscriminator(optimizer=discriminator_optimizer)

    pipeline = Pix2PixPipeline( generator, 
                                discriminator, 
                                generator_loss=generator_loss, 
                                discriminator_loss=discriminator_loss,
                                p2p_loss=pix2pix_loss,
                                lambda_=lambda_)

    for epoch, (raw_image, processed_image) in tqdm(train.repeat().take(epochs).enumerate(), total=epochs, desc='Training Epoch'):
        
        pipeline.train((raw_image, processed_image))
        
        if (epoch == 0) or ((epoch+1) % save_artefact_frequency == 0):
            fig, axes = plt.subplots(4, 4, figsize=(20,20), facecolor='white')
            for (test_raw_image, test_processed_image), ax in zip(test_images, axes.ravel()):
                test_output = tf.concat([generator.predict(test_raw_image, training=True), test_processed_image], axis=2)
                ax.imshow(tf_normalize(test_output[0], (0, 1)))
                ax.axis('off')
            fig.savefig(osp.join(OUTPATH, f'epoch_{epoch+1}_out_grid.png'))
            plt.close(fig)

    pipeline._generator.save_model(osp.join(PIX2PIX_ASSETS, 'saved_model.hdf5'))