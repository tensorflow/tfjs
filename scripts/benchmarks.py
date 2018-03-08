# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Benchmarks for TensorFlow.js Layers.

These benchmarks compare the inference and training speed of Keras models of
varying size and architecture, between Python and browser.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import json
import os
import tempfile
import time

import h5py
import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from scripts import h5_conversion

_FIT_BURNIN_EPOCHS = 1  # How many epochs to call fit() for before timing fit().
_PREDICT_BURNINS = 1  # How many predict() runs to do before timing predict().
_PREDICT_RUNS = 20  # How many runs of predict() to average over.


def benchmark_and_serialize_model(model_fn,
                                  input_shape,
                                  target_shape,
                                  optimizer,
                                  loss,
                                  batch_size,
                                  train_epochs):
  """Benchmark a model's fit() and predict() calls; serialize the model.

  Args:
    model_fn: A function that takes two arguments: `input_shape` and
      `target_shape`, and returns a `keras.Model` instance. The model does not
      need to have been compiled.
    input_shape: Input shape as a `list` or `tuple` of integers.
    target_shape: Target shape as a `list` or `tuple` of integers.
    optimizer: The optimizer to use during training.
    loss: The loss function to use during training.
    batch_size: Batch size to use for training.
    train_epochs: Number of training epochs, not including the burn-in epoch(s).

  Returns:
    1. The model serialized as a JSON string.
    2. The model's trained weights seraizlied as a JSON object.
    3. Total fit() time per epoch, averaged over the epochs not including the
       burn-in one.
    4. Average predict() time over all the _PREDICT_RUNS.
  """
  model = model_fn(input_shape, target_shape)
  model.compile(optimizer=optimizer, loss=loss)
  xs = np.random.rand(*([batch_size] + input_shape))
  ys = np.random.rand(*([batch_size] + target_shape))

  # Perform fit() burn-in.
  model.fit(xs, ys, batch_size=batch_size, epochs=_FIT_BURNIN_EPOCHS)

  # Time fit().
  train_t_begin = time.time()
  model.fit(xs, ys, batch_size=batch_size, epochs=train_epochs)
  train_t_end = time.time()

  # Perform predict() burn-in.
  for _ in range(_PREDICT_BURNINS):
    model.predict(xs)
  # Time predict() by averaging.
  predict_t_begin = time.time()
  for _ in range(_PREDICT_RUNS):
    model.predict(xs)
  predict_t_end = time.time()

  weights_h5_path = tempfile.mktemp() + '.h5'
  model.save_weights(weights_h5_path)
  weights_json = h5_conversion.HDF5Converter().h5_weights_to_json(h5py.File(
      weights_h5_path))
  os.remove(weights_h5_path)

  return (model.to_json(), weights_json,
          (train_t_end - train_t_begin) / train_epochs,
          (predict_t_end - predict_t_begin) / _PREDICT_RUNS)


def dense_tiny_model_fn(input_shape, target_shape):
  assert len(target_shape) == 1
  input_layer = keras.Input(input_shape)
  dense_1 = keras.layers.Dense(200, activation='relu')
  dense_2 = keras.layers.Dense(target_shape[0])
  output = dense_2(dense_1(input_layer))
  model = keras.Model(input_layer, output)
  return model


def dense_large_model_fn(input_shape, target_shape):
  assert len(target_shape) == 1
  input_layer = keras.Input(input_shape)
  dense_1 = keras.layers.Dense(4000, activation='relu')
  dense_2 = keras.layers.Dense(1000, activation='relu')
  dense_3 = keras.layers.Dense(500, activation='relu')
  dense_4 = keras.layers.Dense(target_shape[0])
  output = dense_4(dense_3(dense_2(dense_1(input_layer))))
  model = keras.Model(input_layer, output)
  return model


def convolutional_model_fn(num_filters, input_shape, target_shape):
  """2D convolutional model."""
  kernel_size = 3
  pool_size = 2
  assert len(target_shape) == 1
  num_classes = target_shape[0]
  layers = [
      keras.layers.Conv2D(num_filters, kernel_size,
                          padding='valid',
                          input_shape=input_shape),
      keras.layers.Activation('relu'),
      keras.layers.Conv2D(num_filters, kernel_size),
      keras.layers.Activation('relu'),
      keras.layers.MaxPooling2D(pool_size=pool_size),
      keras.layers.Flatten(),
      keras.layers.Dense(128),
      keras.layers.Activation('relu'),
      keras.layers.Dense(num_classes),
      keras.layers.Activation('softmax')
  ]
  model = keras.models.Sequential(layers)
  return model


def main():
  benchmarks = dict()
  benchmarks['metadata'] = {
      'keras_version': keras.__version__,
      'tensorflow_version': tf.__version__,
      'tensorflow_uses_gpu': any(
          'gpu' in d.name.lower() for d in device_lib.list_local_devices())
  }
  benchmarks['config'] = {
      'FIT_BURNIN_EPOCHS': _FIT_BURNIN_EPOCHS,
      'PREDICT_BURNINS': _PREDICT_BURNINS,
      'PREDICT_RUNS': _PREDICT_RUNS
  }
  benchmarks['models'] = []

  # Dense model.
  optimizer = 'sgd'
  loss = 'mean_squared_error'
  batch_size = 128
  train_epochs = 10
  input_shape = [100]
  target_shape = [1]
  names_fns_and_descriptions = [
      ('dense-tiny',
       dense_tiny_model_fn,
       'Input([%d]);Dense(200);Dense(%d)|%s|%s' %
       (input_shape[0], target_shape[0], optimizer, loss)),
      ('dense-large',
       dense_large_model_fn,
       'Input([%d]);Dense(4000);Dense(1000);Dense(500);Dense(%d)|%s|%s' %
       (input_shape[0], target_shape[0], optimizer, loss))]

  for model_name, model_fn, description in names_fns_and_descriptions:
    (model_json, weights_json, train_time, predict_time) = (
        benchmark_and_serialize_model(model_fn,
                                      input_shape,
                                      target_shape,
                                      optimizer,
                                      loss,
                                      batch_size,
                                      train_epochs))
    print('train_time = %g s' % train_time)
    print('predict_time = %g s' % predict_time)
    benchmarks['models'].append({
        'name': model_name,
        'description': description,
        'optimizer': optimizer,
        'loss': loss,
        'model_json': model_json,
        'weights_json': weights_json,
        'input_shape': input_shape,
        'target_shape': target_shape,
        'batch_size': batch_size,
        'train_epochs': train_epochs,
        'train_time': train_time,
        'predict_time': predict_time,
    })

  # Conv2d models.
  optimizer = 'adam'
  loss = 'categorical_crossentropy'
  input_shape = [28, 28, 1]
  target_shape = [10]
  names_fns_and_descriptions = [
      ("convolutional-%dfilters" % num_filters,
       functools.partial(convolutional_model_fn, num_filters),
       'Conv2D(%d,3);Conv2D(%d,3);MaxPooling2D(2);'
       'Flatten();Dense(128);Dense(10)|%s|%s' %
       (num_filters, num_filters, optimizer, loss)) for num_filters in
      (1, 2, 4, 8, 16, 24, 26, 28, 30, 32)]

  for model_name, model_fn, description in names_fns_and_descriptions:
    model_json, weights_json, train_time, predict_time = (
        benchmark_and_serialize_model(model_fn,
                                      input_shape,
                                      target_shape,
                                      optimizer,
                                      loss,
                                      batch_size,
                                      train_epochs))
    print('train_time = %g s' % train_time)
    print('predict_time = %g s' % predict_time)
    benchmarks['models'].append({
        'name': model_name,
        'description': description,
        'optimizer': optimizer,
        'loss': loss,
        'model_json': model_json,
        'weights_json': weights_json,
        'input_shape': input_shape,
        'target_shape': target_shape,
        'batch_size': batch_size,
        'train_epochs': train_epochs,
        'train_time': train_time,
        'predict_time': predict_time,
    })

  with open(FLAGS.js_path, 'wt') as f:
    f.write('const benchmarks = ' + json.dumps(benchmarks) + ';\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Benchmarks demo.')
  parser.add_argument(
      'js_path',
      type=str,
      help='Path where the data .js file will be saved.')

  FLAGS, _ = parser.parse_known_args()
  main()
