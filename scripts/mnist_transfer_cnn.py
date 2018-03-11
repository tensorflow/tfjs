# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Train a simple CNN model for transfer learning in browser with TF.js Layers.

The model architecture in this file is based on the Keras stock example at:
  https://github.com/keras-team/keras/blob/master/examples/mnist_transfer_cnn.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import json

import keras
from keras import backend as K

from scripts import h5_conversion

# Input image dimensions.
IMG_ROWS, IMG_COLS = 28, 28
NUM_CLASSES = 5

INPUT_SHAPE = ((1, IMG_ROWS, IMG_COLS)
               if  K.image_data_format() == 'channels_first'
               else  (IMG_ROWS, IMG_COLS, 1))


def load_mnist_data(gte5_cutoff):
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

  # create two datasets one with digits below 5 and one with 5 and above
  x_train_lt5 = x_train[y_train < 5]
  y_train_lt5 = y_train[y_train < 5]
  x_test_lt5 = x_test[y_test < 5]
  y_test_lt5 = y_test[y_test < 5]

  x_train_gte5 = x_train[y_train >= 5]
  y_train_gte5 = y_train[y_train >= 5] - 5
  x_test_gte5 = x_test[y_test >= 5]
  y_test_gte5 = y_test[y_test >= 5] - 5

  if gte5_cutoff > 0:
    x_train_gte5 = x_train_gte5[:gte5_cutoff, ...]
    y_train_gte5 = y_train_gte5[:gte5_cutoff, ...]
    x_test_gte5 = x_test_gte5[:gte5_cutoff, ...]
    y_test_gte5 = y_test_gte5[:gte5_cutoff, ...]

  return (x_train_lt5, y_train_lt5, x_test_lt5, y_test_lt5,
          x_train_gte5, y_train_gte5, x_test_gte5, y_test_gte5)


def train_model(model,
                optimizer,
                train,
                test,
                num_classes,
                batch_size=128,
                epochs=5):
  """Train or re-train a model using data.

  Args:
    model: `keras.Model` instance to (re-)train.
    optimizer: Name of the optimizer to use.
    train: Training data, of shape (NUM_EXAMPLES, IMG_ROWS, IMG_COLS).
    test: Test data, of shape (NUM_EXAMPLES, IMG_ROWS, IMG_COLS).
    num_classes: Number of classes.
    batch_size: Batch size.
    epochs: How many epochs to train.
  """
  x_train = train[0].reshape((train[0].shape[0],) + INPUT_SHAPE)
  x_test = test[0].reshape((test[0].shape[0],) + INPUT_SHAPE)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255

  # convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(train[1], num_classes)
  y_test = keras.utils.to_categorical(test[1], num_classes)

  model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

  t = datetime.datetime.now()
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
  print('Training time: %s' % (datetime.datetime.now() - t))
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test score:', score[0])
  print('Test accuracy:', score[1])


def write_mnist_examples_to_json_file(x, y, js_path):
  """Write a batch of MNIST examples to a JavaScript (.js) file.

  Args:
    x: A numpy array representing the image data, with shape
      (NUM_EXAMPLES, IMG_ROWS, IMG_COLS).
    y: A numpy array representing the image labels (as integer indices),
      with shape (NUM_EXAMPLES,).
    js_path: Path to the JavaScript file to write to.
  """
  data = []
  num_examples = x.shape[0]
  for i in range(num_examples):
    data.append({'x': x[i, ...].tolist(), 'y': int(y[i])})
  with open(js_path, 'wt') as f:
    f.write(json.dumps(data))


def write_gte5_data(x_train_gte5,
                    y_train_gte5,
                    x_test_gte5,
                    y_test_gte5,
                    gte5_data_path_prefix):
  """Write the transfer-learning data to .js files.

  Args:
    x_train_lt5: x (image) data for training: digits >= 5.
    y_train_lt5: y (label) data for training: digits >= 5.
    x_test_lt5: x (image) data for test (validation): digits >= 5.
    y_test_lt5: y (label) data for test (validation): digits >= 5.
    gte5_data_path_prefix: Path prefix for writing the files. For example,
      if the value is '/tmp/foo', the train and test files will be written
      at '/tmp/foo.train.js' and '/tmp/foo.test.js', respectively.
  """
  gte5_train_path = gte5_data_path_prefix + '.train.json'
  write_mnist_examples_to_json_file(
      x_train_gte5, y_train_gte5, gte5_train_path)
  print('Wrote gte5 training data to: %s' % gte5_train_path)
  gte5_test_path = gte5_data_path_prefix + '.test.json'
  write_mnist_examples_to_json_file(
      x_test_gte5, y_test_gte5, gte5_test_path)
  print('Wrote gte5 test data to: %s' % gte5_test_path)


def train_and_save_model(filters,
                         kernel_size,
                         pool_size,
                         batch_size,
                         epochs,
                         x_train_lt5,
                         y_train_lt5,
                         x_test_lt5,
                         y_test_lt5,
                         artifacts_dir,
                         optimizer='adam'):
  """Train and save MNIST CNN model.

  Args:
    filters: number of filters for convolution layers.
    kernel_size: kernel size for convolution layers.
    pool_size: pooling kernel size for pooling layers.
    batch_size: batch size.
    epochs: number of epochs to train for.
    x_train_lt5: x (image) data for training: digits < 5.
    y_train_lt5: y (label) data for training: digits < 5.
    x_test_lt5: x (image) data for test (validation): digits < 5.
    y_test_lt5: y (label) data for test (validation): digits < 5.
    artifacts_dir: Directory to save the model artifacts (model topology JSON,
      weights and weight manifest) in.
    optimizer: The name of the optimizer to use, as a string.
  """

  feature_layers = [
      keras.layers.Conv2D(filters, kernel_size,
                          padding='valid',
                          input_shape=INPUT_SHAPE),
      keras.layers.Activation('relu'),
      keras.layers.Conv2D(filters, kernel_size),
      keras.layers.Activation('relu'),
      keras.layers.MaxPooling2D(pool_size=pool_size),
      keras.layers.Dropout(0.25),
      keras.layers.Flatten(),
  ]
  classification_layers = [
      keras.layers.Dense(128),
      keras.layers.Activation('relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(NUM_CLASSES),
      keras.layers.Activation('softmax')
  ]
  model = keras.models.Sequential(feature_layers + classification_layers)

  train_model(model,
              optimizer,
              (x_train_lt5, y_train_lt5),
              (x_test_lt5, y_test_lt5),
              NUM_CLASSES, batch_size=batch_size, epochs=epochs)
  h5_conversion.save_model(model, artifacts_dir)


def main():
  (x_train_lt5, y_train_lt5, x_test_lt5, y_test_lt5,
   x_train_gte5, y_train_gte5, x_test_gte5, y_test_gte5) = load_mnist_data(
       FLAGS.gte5_cutoff)

  write_gte5_data(x_train_gte5, y_train_gte5, x_test_gte5, y_test_gte5,
                  FLAGS.gte5_data_path_prefix)

  train_and_save_model(FLAGS.filters,
                       FLAGS.kernel_size,
                       FLAGS.pool_size,
                       FLAGS.batch_size,
                       FLAGS.epochs,
                       x_train_lt5,
                       y_train_lt5,
                       x_test_lt5,
                       y_test_lt5,
                       FLAGS.artifacts_dir,
                       optimizer=FLAGS.optimizer)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('MNIST model training and serialization')
  parser.add_argument(
      '--epochs',
      type=int,
      default=5,
      help='Number of epochs to train the Keras model for.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Batch size for training.')
  parser.add_argument(
      '--filters',
      type=int,
      default=32,
      help='Number of convolutional filters to use.')
  parser.add_argument(
      '--pool_size',
      type=int,
      default=2,
      help='Size of pooling area for max pooling.')
  parser.add_argument(
      '--kernel_size',
      type=int,
      default=3,
      help='Convolutional kernel size.')
  parser.add_argument(
      '--artifacts_dir',
      type=str,
      default='/tmp/mnist.keras',
      help='Local path for saving the TensorFlow.js artifacts.')
  parser.add_argument(
      '--optimizer',
      type=str,
      default='adam',
      help='Name of the optimizer to use for training.')
  parser.add_argument(
      '--gte5_cutoff',
      type=int,
      default=1024,
      help='If value is > 0, will cause only the first this many examples '
      'in the gte5 (label >= 5) transfer learning subset to be written to '
      'JavaScript (.js) files.')
  parser.add_argument(
      '--gte5_data_path_prefix',
      type=str,
      default='/tmp/mnist_transfer_cnn.gte5',
      help='Prefix for the label >= 5 data for transfer learning.'
      'For example, if the prefix is /tmp/foo.gte5, the train and test '
      'data will be written to /tmp/foo.gte5.train.js and '
      '/tmp/foo.gte5.test.js, respectively.')
  # TODO(cais, soergel): Eventually we want to use the dataset API and not write
  #   writing the data to file.

  FLAGS, _ = parser.parse_known_args()
  main()
