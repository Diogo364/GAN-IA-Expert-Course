from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras.losses import Loss

class WassersteinGeneratorLoss(Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        return -1 * tf.math.reduce_mean(y_pred)


class GradientPenaltyLossInterface(Loss, ABC):
    @abstractmethod
    def gradient_penalty(self):
        pass

class WassersteinDiscriminatorLoss(GradientPenaltyLossInterface):
    def __init__(self, gradient_penalty=None, c_lambda=10):
        self._gradient_penalty = gradient_penalty
        self._c_lambda = c_lambda
        super().__init__()

    def call(self, y_true, y_pred):
        assert self.gradient_penalty is None, 'Must set a value for Gradient Penalty first'
        return tf.cast(tf.math.reduce_mean(y_pred), dtype='float64') - tf.cast(tf.math.reduce_mean(y_true), dtype='float64') + self._c_lambda * self._gradient_penalty

    @property
    def gradient_penalty(self):
        pass

    @gradient_penalty.setter
    def gradient_penalty(self, value):
        self._gradient_penalty = value

class MeanAbsoluteErrorLoss(Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))