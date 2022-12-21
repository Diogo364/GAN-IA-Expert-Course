from abc import ABC, abstractmethod
from utils import remove_extension_names
import os.path as osp
import tensorflow as tf
import tensorflow_datasets as tfds

class DataLoaderInterface(ABC):
    @abstractmethod
    def load_data(self):
        pass


class MNISTDataLoader(DataLoaderInterface):
    def load_data(self):
        return tf.keras.datasets.mnist.load_data()

class FashionMNISTDataLoader(DataLoaderInterface):
    def load_data(self):
        return tf.keras.datasets.fashion_mnist.load_data()

class AbstractURLDataLoader(DataLoaderInterface, ABC):
    def __init__(self, url: str, filename: str=None, extract: bool=True, train_dir: str='train/', test_dir: str='test/', file_extension='jpg'):
        self._url = url
        self._extract = extract
        self._filename = filename if filename is not None else osp.basename(url)
        self._file_extension = file_extension
        self._train_dir = train_dir
        self._test_dir = test_dir
        
    def load_data(self):
        downloaded_data = tf.keras.utils.get_file(fname=self._filename, extract=self._extract, origin=self._url)
        data_path = osp.join(osp.dirname(downloaded_data), remove_extension_names(self._filename))
        train_files = tf.data.Dataset.list_files(osp.join(data_path, f'{self._train_dir}/*.{self._file_extension}'))
        test_files = tf.data.Dataset.list_files(osp.join(data_path, f'{self._test_dir}/*.{self._file_extension}'))
        return train_files, test_files
        

class MapsURLDataLoader(AbstractURLDataLoader):
    def __init__(self, url: str, filename: str=None, extract: bool=True, train_dir: str='train', test_dir: str='val', file_extension='jpg', image_shape=(256, 512)):
        self._image_shape = image_shape
        super().__init__(url, filename, extract, train_dir, test_dir, file_extension)
    
    def _separate_original_target(self, image_path):
    
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img)
        img = tf.image.resize(img, (self._image_shape))
        width = img.shape[1]
        original_img = img[:, :width//2, :]
        target_img = img[:, width//2:, :]
        return tf.cast(original_img, tf.float32), tf.cast(target_img, tf.float32)

    def load_data(self):
        train_files, test_files = super().load_data()
        train_images =  train_files.map(self._separate_original_target, num_parallel_calls=tf.data.AUTOTUNE)
        test_images =  test_files.map(self._separate_original_target, num_parallel_calls=tf.data.AUTOTUNE)
        return (train_images, test_images)

class ApplesNOrangesDataLoader():
    def __init__(self, image_shape=(256, 256, 3)):
        self._image_shape = image_shape
        
    
    @staticmethod
    def _get_image_from_dataset_with_label(image, label):
        return image

    def _reshape_tensor(self, x, y):
        return tf.reshape(x, self._image_shape), tf.reshape(y, self._image_shape)

    def load_data(self):
        dataset = tfds.load('cycle_gan/apple2orange', with_info=False, as_supervised=True)
        train_A, train_B = dataset['trainA'], dataset['trainB']
        test_A, test_B = dataset['testA'], dataset['testB']
        
        train_A, train_B = train_A.map(self._get_image_from_dataset_with_label), train_B.map(self._get_image_from_dataset_with_label)
        test_A, test_B = test_A.map(self._get_image_from_dataset_with_label), test_B.map(self._get_image_from_dataset_with_label)

        train = tf.data.Dataset.zip((train_A, train_B)).map(self._reshape_tensor)
        test = tf.data.Dataset.zip((test_A, test_B)).map(self._reshape_tensor)
        return train, test
