import tensorflow as tf


def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])


def batch_format(element):
    return (tf.reshape(element['pixels'], [-1, 784]),
            tf.reshape(element['label'], [-1, 1]))

