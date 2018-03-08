# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Train a CNN model for MNIST dataset; Export the model and weights.

The model architecture in this file is based on the Keras stock example at:
  https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os

import h5py
import keras
from keras import backend as K

from scripts import h5_conversion

# Input image dimensions.
IMG_ROWS, IMG_COLS = 28, 28
NUM_CLASSES = 10


def _prep_dataset():
  """Prepare MNIST dataset."""
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, IMG_ROWS, IMG_COLS)
    x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)
    input_shape = (1, IMG_ROWS, IMG_COLS)
  else:
    x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
    input_shape = (IMG_ROWS, IMG_COLS, 1)
  y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
  y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')
  return input_shape, x_train, y_train, x_test, y_test


def train(input_shape,
          x_train,
          y_train,
          x_test,
          y_test,
          epochs,
          batch_size,
          model_json_path,
          weights_json_path,
          merged_json_path):
  """Train a Keras model for MNIST data classification and save result as JSON.

  Args:
    input_shape: Shape of inputs, as a tuple of `int`s.
    x_train: Training image data, an `numpy.ndarray` of shape
      `[num_train_examples] + input_shape`.
    y_train: Test image data, an `numpy.ndarray` of shape
      `[num_train_examples, NUM_CLASSES]`.
    x_test: Training target one-hot labels, an `numpy.ndarray` of shape
      `[num_test_examples] + input_shape`.
    y_test: Test target one-hot labels, an `numpy.ndarray` of shape
      `[num_test_examples, NUM_CLASSES]`.
    epochs: Number of epochs to train the Keras model for.
    model_json_path: Path to save the JSON configuration of the trained Keras
      model at.
    weights_json_path: Path to save the JSON serialization of the weights in the
      trained Keras model at.
    merged_json_path: Path to save the architecture-weights merged JSON file.
  """
  model = keras.models.Sequential()
  model.add(keras.layers.Conv2D(
      32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
  model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(keras.layers.Dropout(0.25))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(128, activation='relu'))
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

  merged_model_h5_path = merged_json_path + '.h5'
  keras.models.save_model(model, merged_model_h5_path)
  with open(merged_json_path, 'wt') as f:
    f.write(json.dumps(
        h5_conversion.HDF5Converter().h5_merged_saved_model_to_json(
            merged_model_h5_path)))
  os.remove(merged_model_h5_path)
  print('Saved save_model (model + weights) at %s' % merged_json_path)

  with open(model_json_path, 'wt') as f:
    f.write(model.to_json())
  print('Saved topology at: %s' % model_json_path)

  weights_h5_path = weights_json_path + '.h5'
  model.save_weights(weights_h5_path)
  with open(weights_json_path, 'wt') as f:
    f.write(
        json.dumps(
            h5_conversion.HDF5Converter().h5_weights_to_json(h5py.File(
                weights_h5_path))))
  os.remove(weights_h5_path)
  print('Saved weights at: %s' % weights_json_path)


def main():
  input_shape, x_train, y_train, x_test, y_test = _prep_dataset()
  train(input_shape, x_train, y_train, x_test, y_test,
        FLAGS.epochs, FLAGS.batch_size, FLAGS.model_json_path,
        FLAGS.weights_json_path, FLAGS.merged_json_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('MNIST model training and serialization')
  parser.add_argument(
      '--epochs',
      type=int,
      default=12,
      help='Number of epochs to train the Keras model for.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Batch size for training.')
  parser.add_argument(
      '--model_json_path',
      type=str,
      default='/tmp/mnist.keras.model.json',
      help='Local path for the Keras model definition JSON file.')
  parser.add_argument(
      '--weights_json_path',
      type=str,
      default='/tmp/mnist.keras.weights.json',
      help='Local path for the Keras model weights JSON file.')
  parser.add_argument(
      '--merged_json_path',
      type=str,
      default='/tmp/mnist.keras.merged.json',
      help='Local path for the Keras model topology & weights JSON file.')

  FLAGS, _ = parser.parse_known_args()
  main()
