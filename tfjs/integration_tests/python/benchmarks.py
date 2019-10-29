# @license
# Copyright 2019 Google LLC. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Benchmarks for TensorFlow.js tfjs-layers and tfjs-node.

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
import subprocess
import shutil
import sys
import tempfile
import time

from tensorflow import keras
import numpy as np
import tensorflow as tf
# Comparing TF Eager vs TF.js for a fair comparison.
if hasattr(tf, 'enable_eager_execution'):
  tf.enable_eager_execution()
from tensorflow.python.client import device_lib
import tensorflowjs as tfjs

_FIT_BURNIN_EPOCHS = 1  # How many epochs to call fit() for before timing fit().
_PREDICT_BURNINS = 20  # How many predict() runs to do before timing predict().
_PREDICT_RUNS = 30  # How many runs of predict() to average over.


def _is_gpu_available():
  devices = device_lib.list_local_devices()
  for device in devices:
    if device.device_type == 'GPU':
      return True
  return False


def _get_random_inputs_and_outputs(model, batch_size):
  """Synthesize random inputs and outputs based on the model's specs.

  Args:
    model: An instance of keras Model.
    batch_size: Desired batch size.

  Returns:
    xs: Synthesized random feature tensor(s).
    ys: Synthesized random target tensor(s).
  """
  input_shapes = [[
      int(d) for d in list(inp.shape[1:])] for inp in model.inputs]
  xs = []
  for in_shape in input_shapes:
    x = np.random.rand(*([batch_size] + in_shape))
    xs.append(x)
  if len(xs) == 1:
    xs = xs[0]

  output_shapes = [[
      int(d) for d in list(inp.shape[1:])] for inp in model.outputs]
  ys = []
  for output_shape in output_shapes:
    y = np.random.rand(*([batch_size] + output_shape))
    ys.append(y)
  if len(ys) == 1:
    ys = ys[0]

  return xs, ys


def benchmark_and_serialize_model(model_name,
                                  description,
                                  model_fn,
                                  input_shape,
                                  target_shape,
                                  optimizer,
                                  loss,
                                  batch_size,
                                  train_epochs,
                                  artifacts_dir,
                                  export_saved_model=False):
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
    artifacts_dir: Directory to save the data in. The data includes:
      * topology and weights of the models, in TensorFlow.js format
      * metadata and benchmark information in a file named `data.json`,
        including:
        - the name and description of the model
        - the name of the optimizer used during benchmarking of training
        - loss value
        - the input and output shapes of the model
        - benchmark results from Python Keras.

  Returns:
    1. Total fit() time per epoch, averaged over the epochs not including the
       burn-in one.
    2. Average predict() time over all the _PREDICT_RUNS.
  """
  if os.path.isdir(artifacts_dir) and os.listdir(artifacts_dir):
    for rel_name in os.listdir(artifacts_dir):
      abs_path = os.path.join(artifacts_dir, rel_name)
      if os.path.isfile(abs_path):
        os.remove(abs_path)
      else:
        shutil.rmtree(abs_path)

  environment_info = _get_python_environment_info()
  task_logs = {}

  model = model_fn(input_shape, target_shape)
  if train_epochs:
    model.compile(optimizer=optimizer, loss=loss)

  xs, ys = _get_random_inputs_and_outputs(model, batch_size)

  # Perform fit() burn-in.
  if train_epochs:
    model.fit(xs, ys, batch_size=batch_size, epochs=_FIT_BURNIN_EPOCHS)

  # Time fit().
  if train_epochs:
    train_t_begin = time.time()
    model.fit(xs, ys, batch_size=batch_size, epochs=train_epochs)
    train_t_end = time.time()

  # Save data about the model and benchmark results.
  if train_epochs:
    train_time = (train_t_end - train_t_begin) * 1e3 / train_epochs

    # Collect and format the data for fit().
    task_logs['fit'] = {  # For schema, see 'ModelTrainingBenchmarkRun` in types.ts.
      'taskType': 'model',
      'modelFormat': 'GraphModel' if export_saved_model else 'LayersModel',
      'modelName': model_name,
      'modelDescription': description,
      'functionName': 'fit',
      'endingTimestampMs': int(time.time() * 1e3),
      'batchSize': batch_size,
      'optimizer': optimizer.__class__.__name__.split('.')[-1],
      'loss': loss,
      'numBenchmarkedIterations': train_epochs,
      'numWarmUpIterations': _FIT_BURNIN_EPOCHS,
      'averageTimeMs': train_time
    }

  # Perform predict() burn-in.
  for _ in range(_PREDICT_BURNINS):
    model.predict(xs)
  # Time predict() by averaging.
  predict_ts = []
  for _ in range(_PREDICT_RUNS):
    predict_t_begin = time.time()
    model.predict(xs)
    predict_ts.append((time.time() - predict_t_begin) * 1e3)

  if export_saved_model:
    tmp_saved_model_dir = tempfile.mkdtemp()
    tf.compat.v1.keras.experimental.export_saved_model(
        model, tmp_saved_model_dir, serving_only=True)
    subprocess.check_output([
        'tensorflowjs_converter',
        '--input_format', 'tf_saved_model',
        '--output_format', 'tfjs_graph_model',
        '--signature_name', 'serving_default',
        '--saved_model_tags', 'serve',
        tmp_saved_model_dir, artifacts_dir])
    # Clean up the temporary SavedModel directory.
    shutil.rmtree(tmp_saved_model_dir)
  else:
    # Save the model and weights.
    tfjs.converters.save_keras_model(model, artifacts_dir)

  # Collect and format the data for predict().
  task_logs['predict'] = {  # For schema, see 'ModelTaskLog` in types.ts.
    'taskType': 'model',
    'modelFormat': 'GraphModel' if export_saved_model else 'LayersModel',
    'modelName': model_name,
    'modelDescription': description,
    'functionName': 'predict',
    'endingTimestampMs': int(time.time() * 1e3),
    'batchSize': batch_size,
    'numBenchmarkedIterations': _PREDICT_RUNS,
    'numWarmUpIterations': _PREDICT_BURNINS,
    'timesMs': predict_ts,
    'averageTimeMs': np.mean(predict_ts)
  }

  return task_logs


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


def mobilenet_v2_model_fn(alpha, input_shape, target_shape):
  """MobileNetV2: A ConvNet from Keras Applications."""
  del input_shape, target_shape  # Unused.
  # `weights=None` leads to random weight initialization and downloadnig
  # of weights.
  model = keras.applications.MobileNetV2(alpha=alpha, weights=None)
  return model


def attention_model_fn(input_shape, target_shape):
  """Attention-based translation model."""
  del input_shape, target_shape  # Unused.
  model_json = '{"class_name":"Model","config":{"input_layers":[["input_1",0,0],["s0",0,0],["c0",0,0]],"name":"model_1","layers":[{"class_name":"InputLayer","inbound_nodes":[],"name":"input_1","config":{"dtype":"float32","name":"input_1","sparse":false,"batch_input_shape":[null,30,38]}},{"class_name":"InputLayer","inbound_nodes":[],"name":"s0","config":{"dtype":"float32","name":"s0","sparse":false,"batch_input_shape":[null,64]}},{"class_name":"Bidirectional","inbound_nodes":[[["input_1",0,0,{}]]],"name":"bidirectional_1","config":{"trainable":true,"name":"bidirectional_1","merge_mode":"concat","layer":{"class_name":"LSTM","config":{"stateful":false,"units":32,"activation":"tanh","recurrent_activation":"hard_sigmoid","dropout":0,"recurrent_dropout":0,"use_bias":true,"trainable":true,"recurrent_initializer":{"class_name":"Orthogonal","config":{"seed":null,"gain":1}},"bias_constraint":null,"unroll":false,"kernel_initializer":{"class_name":"VarianceScaling","config":{"seed":null,"distribution":"uniform","mode":"fan_avg","scale":1}},"unit_forget_bias":true,"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_constraint":null,"activity_regularizer":null,"return_sequences":true,"recurrent_constraint":null,"recurrent_regularizer":null,"bias_regularizer":null,"go_backwards":false,"implementation":1,"name":"attLSTM_2","kernel_regularizer":null,"return_state":false}}}},{"class_name":"RepeatVector","inbound_nodes":[[["s0",0,0,{}]],[["attLSTM_1",0,0,{}]],[["attLSTM_1",1,0,{}]],[["attLSTM_1",2,0,{}]],[["attLSTM_1",3,0,{}]],[["attLSTM_1",4,0,{}]],[["attLSTM_1",5,0,{}]],[["attLSTM_1",6,0,{}]],[["attLSTM_1",7,0,{}]],[["attLSTM_1",8,0,{}]]],"name":"repeat_vector_1","config":{"n":30,"trainable":true,"name":"repeat_vector_1"}},{"class_name":"Concatenate","inbound_nodes":[[["bidirectional_1",0,0,{}],["repeat_vector_1",0,0,{}]],[["bidirectional_1",0,0,{}],["repeat_vector_1",1,0,{}]],[["bidirectional_1",0,0,{}],["repeat_vector_1",2,0,{}]],[["bidirectional_1",0,0,{}],["repeat_vector_1",3,0,{}]],[["bidirectional_1",0,0,{}],["repeat_vector_1",4,0,{}]],[["bidirectional_1",0,0,{}],["repeat_vector_1",5,0,{}]],[["bidirectional_1",0,0,{}],["repeat_vector_1",6,0,{}]],[["bidirectional_1",0,0,{}],["repeat_vector_1",7,0,{}]],[["bidirectional_1",0,0,{}],["repeat_vector_1",8,0,{}]],[["bidirectional_1",0,0,{}],["repeat_vector_1",9,0,{}]]],"name":"concatenate_1","config":{"trainable":true,"name":"concatenate_1","axis":-1}},{"class_name":"Dense","inbound_nodes":[[["concatenate_1",0,0,{}]],[["concatenate_1",1,0,{}]],[["concatenate_1",2,0,{}]],[["concatenate_1",3,0,{}]],[["concatenate_1",4,0,{}]],[["concatenate_1",5,0,{}]],[["concatenate_1",6,0,{}]],[["concatenate_1",7,0,{}]],[["concatenate_1",8,0,{}]],[["concatenate_1",9,0,{}]]],"name":"attDense_1","config":{"bias_constraint":null,"kernel_constraint":null,"units":10,"activity_regularizer":null,"use_bias":true,"bias_regularizer":null,"trainable":true,"activation":"tanh","name":"attDense_1","kernel_initializer":{"class_name":"VarianceScaling","config":{"seed":null,"distribution":"uniform","mode":"fan_avg","scale":1}},"kernel_regularizer":null,"bias_initializer":{"class_name":"Zeros","config":{}}}},{"class_name":"Dense","inbound_nodes":[[["attDense_1",0,0,{}]],[["attDense_1",1,0,{}]],[["attDense_1",2,0,{}]],[["attDense_1",3,0,{}]],[["attDense_1",4,0,{}]],[["attDense_1",5,0,{}]],[["attDense_1",6,0,{}]],[["attDense_1",7,0,{}]],[["attDense_1",8,0,{}]],[["attDense_1",9,0,{}]]],"name":"attDense_2","config":{"bias_constraint":null,"kernel_constraint":null,"units":1,"activity_regularizer":null,"use_bias":true,"bias_regularizer":null,"trainable":true,"activation":"relu","name":"attDense_2","kernel_initializer":{"class_name":"VarianceScaling","config":{"seed":null,"distribution":"uniform","mode":"fan_avg","scale":1}},"kernel_regularizer":null,"bias_initializer":{"class_name":"Zeros","config":{}}}},{"class_name":"Activation","inbound_nodes":[[["attDense_2",0,0,{}]],[["attDense_2",1,0,{}]],[["attDense_2",2,0,{}]],[["attDense_2",3,0,{}]],[["attDense_2",4,0,{}]],[["attDense_2",5,0,{}]],[["attDense_2",6,0,{}]],[["attDense_2",7,0,{}]],[["attDense_2",8,0,{}]],[["attDense_2",9,0,{}]]],"name":"attention_weights","config":{"trainable":true,"activation":"softmax","name":"attention_weights"}},{"class_name":"Dot","inbound_nodes":[[["attention_weights",0,0,{}],["bidirectional_1",0,0,{}]],[["attention_weights",1,0,{}],["bidirectional_1",0,0,{}]],[["attention_weights",2,0,{}],["bidirectional_1",0,0,{}]],[["attention_weights",3,0,{}],["bidirectional_1",0,0,{}]],[["attention_weights",4,0,{}],["bidirectional_1",0,0,{}]],[["attention_weights",5,0,{}],["bidirectional_1",0,0,{}]],[["attention_weights",6,0,{}],["bidirectional_1",0,0,{}]],[["attention_weights",7,0,{}],["bidirectional_1",0,0,{}]],[["attention_weights",8,0,{}],["bidirectional_1",0,0,{}]],[["attention_weights",9,0,{}],["bidirectional_1",0,0,{}]]],"name":"dot_1","config":{"trainable":true,"name":"dot_1","normalize":false,"axes":1}},{"class_name":"InputLayer","inbound_nodes":[],"name":"c0","config":{"dtype":"float32","name":"c0","sparse":false,"batch_input_shape":[null,64]}},{"class_name":"LSTM","inbound_nodes":[[["dot_1",0,0,{}],["s0",0,0,{}],["c0",0,0,{}]],[["dot_1",1,0,{}],["attLSTM_1",0,0,{}],["attLSTM_1",0,2,{}]],[["dot_1",2,0,{}],["attLSTM_1",1,0,{}],["attLSTM_1",1,2,{}]],[["dot_1",3,0,{}],["attLSTM_1",2,0,{}],["attLSTM_1",2,2,{}]],[["dot_1",4,0,{}],["attLSTM_1",3,0,{}],["attLSTM_1",3,2,{}]],[["dot_1",5,0,{}],["attLSTM_1",4,0,{}],["attLSTM_1",4,2,{}]],[["dot_1",6,0,{}],["attLSTM_1",5,0,{}],["attLSTM_1",5,2,{}]],[["dot_1",7,0,{}],["attLSTM_1",6,0,{}],["attLSTM_1",6,2,{}]],[["dot_1",8,0,{}],["attLSTM_1",7,0,{}],["attLSTM_1",7,2,{}]],[["dot_1",9,0,{}],["attLSTM_1",8,0,{}],["attLSTM_1",8,2,{}]]],"name":"attLSTM_1","config":{"stateful":false,"units":64,"activation":"tanh","recurrent_activation":"hard_sigmoid","dropout":0,"recurrent_dropout":0,"use_bias":true,"trainable":true,"recurrent_initializer":{"class_name":"Orthogonal","config":{"seed":null,"gain":1}},"bias_constraint":null,"unroll":false,"kernel_initializer":{"class_name":"VarianceScaling","config":{"seed":null,"distribution":"uniform","mode":"fan_avg","scale":1}},"unit_forget_bias":true,"bias_initializer":{"class_name":"Zeros","config":{}},"kernel_constraint":null,"activity_regularizer":null,"return_sequences":false,"recurrent_constraint":null,"recurrent_regularizer":null,"bias_regularizer":null,"go_backwards":false,"implementation":1,"name":"attLSTM_1","kernel_regularizer":null,"return_state":true}},{"class_name":"Dense","inbound_nodes":[[["attLSTM_1",0,0,{}]],[["attLSTM_1",1,0,{}]],[["attLSTM_1",2,0,{}]],[["attLSTM_1",3,0,{}]],[["attLSTM_1",4,0,{}]],[["attLSTM_1",5,0,{}]],[["attLSTM_1",6,0,{}]],[["attLSTM_1",7,0,{}]],[["attLSTM_1",8,0,{}]],[["attLSTM_1",9,0,{}]]],"name":"attDense_3","config":{"bias_constraint":null,"kernel_constraint":null,"units":11,"activity_regularizer":null,"use_bias":true,"bias_regularizer":null,"trainable":true,"activation":"softmax","name":"attDense_3","kernel_initializer":{"class_name":"VarianceScaling","config":{"seed":null,"distribution":"uniform","mode":"fan_avg","scale":1}},"kernel_regularizer":null,"bias_initializer":{"class_name":"Zeros","config":{}}}}],"output_layers":[["attDense_3",0,0],["attDense_3",1,0],["attDense_3",2,0],["attDense_3",3,0],["attDense_3",4,0],["attDense_3",5,0],["attDense_3",6,0],["attDense_3",7,0],["attDense_3",8,0],["attDense_3",9,0]]}}';
  model = keras.models.model_from_json(model_json)
  return model


_RNN_TYPE_MAP = {
    'SimpleRNN': keras.layers.SimpleRNN,
    'GRU': keras.layers.GRU,
    'LSTM': keras.layers.LSTM
}


def rnn_model_fn(rnn_type, input_shape, target_shape):
  """Recurrent neural network model."""
  rnnConstructor = _RNN_TYPE_MAP[rnn_type]
  layers = [rnnConstructor(target_shape[0], input_shape=input_shape)]
  model = keras.models.Sequential(layers)
  return model


def _get_environment_type():
  return ('python-tensorflow-cuda' if tf.test.gpu_device_name() else
          'python-tensorflow-cpu')


def _get_python_environment_info():
  environment_info = {  # For schema, see `PythonEnvironmentInfo` in types.ts.
    'type': _get_environment_type()
  }

  try:
    # Disable color from `inxi` command.
    environment_info['cpuInfo'] = tf.compat.as_str(
        subprocess.check_output(['inxi', '-c', '0']))
  except:
    pass
  try:
    environment_info['memInfo'] = tf.compat.as_str(
        subprocess.check_output(['free']))
  except:
    pass
  try:
    environment_info['systemInfo'] = tf.compat.as_str(
        subprocess.check_output(['uname', '-a']))
  except:
    pass

  python_ver = sys.version_info
  environment_info['pythonVersion'] = '%d.%d.%d' % (
      python_ver.major, python_ver.minor, python_ver.micro)
  environment_info['tensorflowVersion'] = tf.__version__
  environment_info['kerasVersion'] = keras.__version__
  return environment_info


def main():
  environment_info = _get_python_environment_info()
  print('Environment info:')
  print(json.dumps(environment_info, indent=2))

  suite_log = dict()  # For schema, see `SuiteLog` in types.ts.
  suite_log['data'] = {}
  suite_log['environmentInfo'] = environment_info

  # Dense model.
  optimizer = tf.keras.optimizers.SGD()
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
    suite_log['data'][model_name] = (
        benchmark_and_serialize_model(
            model_name,
            description,
            model_fn,
            input_shape,
            target_shape,
            optimizer,
            loss,
            batch_size,
            train_epochs,
            os.path.join(FLAGS.data_root, model_name)))

  # dense-tiny and dense-large models as TensorFlow SavedModel (inference only;
  # CPU only).
  # TODO(cais): Make this run on tfjs-node-gpu as well.
  if not _is_gpu_available():
    optimizer = None
    loss = None
    batch_size = 128
    train_epochs = 0
    input_shape = [100]
    target_shape = [1]
    names_fns_and_descriptions = [
        ('dense-tiny_GraphModel',
        dense_tiny_model_fn,
        'Input([%d]);Dense(200);Dense(%d)|%s|%s' %
        (input_shape[0], target_shape[0], optimizer, loss)),
        ('dense-large_GraphModel',
        dense_large_model_fn,
        'Input([%d]);Dense(4000);Dense(1000);Dense(500);Dense(%d)|%s|%s' %
        (input_shape[0], target_shape[0], optimizer, loss))]

    for model_name, model_fn, description in names_fns_and_descriptions:
      suite_log['data'][model_name] = (
          benchmark_and_serialize_model(
              model_name,
              description,
              model_fn,
              input_shape,
              target_shape,
              optimizer,
              loss,
              batch_size,
              train_epochs,
              os.path.join(FLAGS.data_root, model_name),
              export_saved_model=True))

  # Conv2d models.
  optimizer = tf.keras.optimizers.SGD()
  loss = 'categorical_crossentropy'
  train_epochs = 10
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
    suite_log['data'][model_name] = (
        benchmark_and_serialize_model(
            model_name,
            description,
            model_fn,
            input_shape,
            target_shape,
            optimizer,
            loss,
            batch_size,
            train_epochs,
            os.path.join(FLAGS.data_root, model_name)))

  # RNN models.
  # TODO(cais): Restore optimizer after the following
  #   error is resolved:
  # "Error: Cannot evaluate flag 'EPSILON': no evaluation function found."
  # optimizer = tf.keras.optimizers.RMSProp()
  # loss = 'categorical_crossentropy'
  # train_epochs = 10
  optimizer = None
  loss = None
  train_epochs = 0
  input_shape = [20, 20]
  target_shape = [20]
  batch_size = 128
  names_fns_and_descriptions = [
      ("rnn-%s" % rnn_type,
       functools.partial(rnn_model_fn, rnn_type),
       '%s(input_shape=%s, target_shape=%s)|%s|%s' %
       (rnn_type, input_shape, target_shape, optimizer, loss))
      for rnn_type in ('SimpleRNN', 'GRU', 'LSTM')]

  for model_name, model_fn, description in names_fns_and_descriptions:
    suite_log['data'][model_name] = (
        benchmark_and_serialize_model(
            model_name,
            description,
            model_fn,
            input_shape,
            target_shape,
            optimizer,
            loss,
            batch_size,
            train_epochs,
            os.path.join(FLAGS.data_root, model_name)))

  # MobileNetV2 (inference only).
  input_shape = None  # Determine from the Model object itself.
  target_shape = None  # Determine from the Model object itself.
  batch_size = 8
  train_epochs = 0
  optimizer = None
  loss = None
  names_fns_and_descriptions = [[
      'mobilenet_v2_%.3d' % (alpha * 100),
      functools.partial(mobilenet_v2_model_fn, alpha),
      'mobilenet_v2_%.3d' % (alpha * 100)] for alpha in (0.25, 0.5, 0.75, 1)]
  for model_name, model_fn, description in names_fns_and_descriptions:
    suite_log['data'][model_name] = (
        benchmark_and_serialize_model(
            model_name,
            description,
            model_fn,
            input_shape,
            target_shape,
            optimizer,
            loss,
            batch_size,
            train_epochs,
            os.path.join(FLAGS.data_root, model_name)))

  # Attention model (inference only).
  input_shape = None  # Determine from the Model object itself.
  target_shape = None  # Determine from the Model object itself.
  batch_size = 32
  train_epochs = 0
  optimizer = None
  loss = None
  names_fns_and_descriptions = [[
      'attention',
      attention_model_fn,
      'Attention-based translation model: '
      'Function model with bidirectional LSTM layers']]
  for model_name, model_fn, description in names_fns_and_descriptions:
    suite_log['data'][model_name] = (
        benchmark_and_serialize_model(
            model_name,
            description,
            model_fn,
            input_shape,
            target_shape,
            optimizer,
            loss,
            batch_size,
            train_epochs,
            os.path.join(FLAGS.data_root, model_name)))

  # MobileNetV2 as TensorFlow SavedModel (inference only; CPU only).
  if not _is_gpu_available():
    # TODO(cais): Make the benchmark run on tensorflow-gpu as well.
    input_shape = None  # Determine from the Model object itself.
    target_shape = None  # Determine from the Model object itself.
    batch_size = 8
    train_epochs = 0
    optimizer = None
    loss = None
    names_fns_and_descriptions = [[
        'mobilenet_v2_%.3d_GraphModel' % (alpha * 100),
        functools.partial(mobilenet_v2_model_fn, alpha),
        'mobilenet_v2_%.3d_GraphModel' % (alpha * 100)]
        for alpha in (0.25, 0.5, 0.75, 1)]
    for model_name, model_fn, description in names_fns_and_descriptions:
      suite_log['data'][model_name] = (
          benchmark_and_serialize_model(
              model_name,
              description,
              model_fn,
              input_shape,
              target_shape,
              optimizer,
              loss,
              batch_size,
              train_epochs,
              os.path.join(FLAGS.data_root, model_name),
              export_saved_model=True))

  # TODO(cais): Add fitDataset() calls (i.e., equivalent to fit() with a
  #   tf.data.Dataset object i nPython).

  with open(os.path.join(FLAGS.data_root, 'benchmarks.json'), 'wt') as f:
    json.dump(suite_log, f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Benchmarks demo.')
  parser.add_argument(
      'data_root',
      type=str,
      help='Local path for saving the results of benchmarks.')
  parser.add_argument(
      '--hash_converter', type=str, help='Commit hash of tfjs-conveter')
  parser.add_argument(
      '--hash_core', type=str, help='Commit hash of tfjs-core')
  parser.add_argument(
      '--hash_data', type=str, help='Commit hash of tfjs-data')
  parser.add_argument(
      '--hash_layers', type=str, help='Commit hash of tfjs-layers')

  FLAGS, _ = parser.parse_known_args()
  main()
