import tensorflow as tf
from tensorflow.keras.layers import Layer

class ChannelPadding(Layer):
    def __init__(self, padding, mode='CONSTANT', **kwargs):
        super(ChannelPadding, self).__init__(**kwargs)
        self.padding = padding
        self.mode = mode

    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [0, 0], [0, 0], [0, self.padding]], self.mode)

    def compute_output_shape(self, input_shape):
        batch, dim1, dim2, values = input_shape
        return (batch, dim1, dim2, values + self.padding)

    def get_config(self):
        config = {
            'padding': self.padding,
            'mode': self.mode,
        }
        base_config = super(ChannelPadding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class StopGradient(Layer):

  def call(self, inputs, **kwargs):
    return tf.stop_gradient(inputs)
