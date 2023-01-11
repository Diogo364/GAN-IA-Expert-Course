import os
import os.path as osp
from tqdm import tqdm
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from utils import normalize, random_crop, random_flip, load_env, tf_normalize
import matplotlib.pyplot as plt
from models.dataloaders import ApplesNOrangesDataLoader
from models.preprocessor import GenericPreprocessor
from models.nn.gan_pipeline import CycleGANPipeline
from models.nn.baseInterfaces import CustomArchitectureModel
from utils.custom_losses import MeanAbsoluteErrorLoss

load_env()

def get_fruit_data():
    generic_train_preprocessor = GenericPreprocessor(preprocessor_list=[random_crop, random_flip, normalize])
    generic_test_preprocessor = GenericPreprocessor(preprocessor_list=[normalize])
    (train, test) = ApplesNOrangesDataLoader().load_data()
    train = train.map(generic_train_preprocessor)
    test = test.map(generic_test_preprocessor)
    return train, test

if __name__ == '__main__':
    OUTPATH = osp.join(os.environ['OUTPATH'], 'Apple2Orange_CycleGAN')
    CycleGAN_ASSETS = osp.join(os.environ['ASSETS'], 'Apple2Orange_CycleGAN')

    os.makedirs(OUTPATH, exist_ok=True)
    os.makedirs(CycleGAN_ASSETS, exist_ok=True)

    test_size = 16
    batch_size = 1
    buffer_size = 995
    random_noise_size = (256,256,3)
    epochs = 10
    save_artefact_frequency = 2 # each n epochs
    learning_rate = 0.0002
    beta1, beta2 = 0.5, 0.999
    lambda_ = 10

    train, test = get_fruit_data()
    
    train = train.shuffle(buffer_size).batch(batch_size).prefetch(batch_size*2)
    train_size =  len(list(train))
    test = test.batch(batch_size)

    test_images = test.take(test_size)

    generator_A2B_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)
    generator_B2A_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)
    
    discriminator_A2B_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)
    discriminator_B2A_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)

    generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    discriminator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    cycle_loss = MeanAbsoluteErrorLoss()
    identity_loss = MeanAbsoluteErrorLoss()

    generator_A2B = CustomArchitectureModel(model=pix2pix.unet_generator(random_noise_size[-1], norm_type='instancenorm'), 
                                            optimizer=generator_A2B_optimizer,
                                            input_shape=random_noise_size)

    generator_B2A = CustomArchitectureModel(model=pix2pix.unet_generator(random_noise_size[-1], norm_type='instancenorm'), 
                                            optimizer=generator_B2A_optimizer,
                                            input_shape=random_noise_size)

    
    discriminator_A2B = CustomArchitectureModel(model=pix2pix.discriminator(norm_type='instancenorm', target=False), 
                                            optimizer=discriminator_A2B_optimizer,
                                            input_shape=random_noise_size)

    discriminator_B2A = CustomArchitectureModel(model=pix2pix.discriminator(norm_type='instancenorm', target=False), 
                                            optimizer=discriminator_B2A_optimizer,
                                            input_shape=random_noise_size)

    pipeline = CycleGANPipeline(generatorA2B=generator_A2B,
                                generatorB2A=generator_B2A,
                                discriminatorA2B=discriminator_A2B,
                                discriminatorB2A=discriminator_B2A,
                                generator_loss=generator_loss,
                                discriminator_loss=discriminator_loss,
                                cycle_loss=cycle_loss,
                                identity_loss=identity_loss,
                                lambda_=lambda_)

    for epoch in tqdm(range(epochs), leave=True, desc='Training Epoch'):
        for (train_A, train_B) in tqdm(train, total=train_size//batch_size, desc='Batches', leave=False):
            pipeline.train((train_A, train_B))
        if (epoch == 0) or ((epoch+1) % save_artefact_frequency == 0):
            fig, axes = plt.subplots(4, 4, figsize=(20,20), facecolor='white')
            for (test_A, test_B), ax in zip(test_images, axes.ravel()):
                test_A2B = tf.concat([test_A, generator_A2B.predict(test_A)], axis=2)
                test_B2A = tf.concat([test_B, generator_B2A.predict(test_B)], axis=2)
                ax.imshow(
                    tf_normalize(
                        tf.concat([test_A2B, test_B2A], axis=1)[0], 
                        (0, 1)
                    )
                )
                ax.axis('off')
            fig.savefig(osp.join(OUTPATH, f'epoch_{epoch+1}_out_grid.png'))
            plt.close(fig)
            
            pipeline._generator.model.save_weigths(osp.join(CycleGAN_ASSETS, 'saved_model_A2B.hd5'))
            pipeline._generator_reverse.model.save_weigths(osp.join(CycleGAN_ASSETS, 'saved_model_B2A.hd5'))