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

# This file is 1/2 of the test suites for CUJ: convert->predict.
#
# This file does below things:
# - Create saved models with TensorFlow.
# - Convert the saved models to tfjs format and store in files.
# - Store inputs in files.
# - Make inference and store outputs in files.

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

import numpy as np
import tensorflow as tf
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import variables
from tensorflow.python.training.tracking import tracking
from tensorflow.python.saved_model.save import save
import tensorflow_hub as hub
import tensorflowjs as tfjs

curr_dir = os.path.dirname(os.path.realpath(__file__))
_tmp_dir = os.path.join(curr_dir, 'convert_predict_data')

def _save_and_convert_model(model_fn, model_path, control_flow_v2=False):
  """Benchmark a model's fit() and predict() calls; serialize the model.

  Args:
    model_fn: A function that creates the saved model.
    model_path: Path to construct files related to the model.

  Returns:
    predict task_log hash that specifies the inputs and outputs for
    validation test.
  """
  # Generate model, inputs, and outputs using Tensorflow.
  tmp_saved_model_dir = tempfile.mkdtemp()
  model_info = model_fn(tmp_saved_model_dir)

  # Write inputs to file.
  xs_data = []
  xs_shape = []
  xs_dtype = []
  xs_names = []
  keys = model_info['inputs'].keys()
  for key in keys:
    xs_names.append(key)
    xs_data.append(model_info['inputs'][key]['value'])
    xs_shape.append(model_info['inputs'][key]['shape'])
    xs_dtype.append(model_info['inputs'][key]['dtype'])

  xs_name_path = os.path.join(_tmp_dir, model_path + '.xs-name.json')
  xs_shape_path = os.path.join(_tmp_dir, model_path + '.xs-shapes.json')
  xs_data_path = os.path.join(_tmp_dir, model_path + '.xs-data.json')
  xs_dtype_path = os.path.join(_tmp_dir, model_path + '.xs-dtype.json')
  with open(xs_name_path, 'w') as f:
    f.write(json.dumps(xs_names))
  with open(xs_data_path, 'w') as f:
    f.write(json.dumps(xs_data))
  with open(xs_shape_path, 'w') as f:
    f.write(json.dumps(xs_shape))
  with open(xs_dtype_path, 'w') as f:
    f.write(json.dumps(xs_dtype))
  # Write outputs to file.
  ys_data = []
  ys_shape = []
  ys_dtype = []
  ys_names = []
  keys = model_info['outputs'].keys()
  for key in keys:
    ys_names.append(key)
    ys_data.append(model_info['outputs'][key]['value'])
    ys_shape.append(model_info['outputs'][key]['shape'])
    ys_dtype.append(model_info['outputs'][key]['dtype'])

  ys_name_path = os.path.join(_tmp_dir, model_path + '.ys-name.json')
  ys_data_path = os.path.join(_tmp_dir, model_path + '.ys-data.json')
  ys_shape_path = os.path.join(_tmp_dir, model_path + '.ys-shapes.json')
  ys_dtype_path = os.path.join(_tmp_dir, model_path + '.ys-dtype.json')
  with open(ys_name_path, 'w') as f:
    f.write(json.dumps(ys_names))
  with open(ys_data_path, 'w') as f:
    f.write(json.dumps(ys_data))
  with open(ys_shape_path, 'w') as f:
    f.write(json.dumps(ys_shape))
  with open(ys_dtype_path, 'w') as f:
    f.write(json.dumps(ys_dtype))
  artifacts_dir = os.path.join(_tmp_dir, model_path)

  # Convert and store model to file.
  args = [
      'tensorflowjs_converter',
      '--input_format', 'tf_saved_model',
      '--output_format', 'tfjs_graph_model',
      '--signature_name', 'serving_default',
      '--saved_model_tags', 'serve'];
  if control_flow_v2:
    args = args + ['--control_flow_v2', 'True']

  print(args, tmp_saved_model_dir, artifacts_dir)
  subprocess.check_output(args +[tmp_saved_model_dir, artifacts_dir])

def _create_saved_model_v1(save_dir):
  """Create a TensorFlow V1 SavedModel for testing.

  Args:
    save_dir: directory name of where the saved model will be stored.
  """

  graph = tf.Graph()
  with graph.as_default():
    input = tf.compat.v1.placeholder(tf.float32, shape=[2, 2])
    w = tf.compat.v1.get_variable('w', shape=[2, 2])
    output = tf.compat.v1.matmul(input, w)
    init_op = w.initializer

    # Create a builder.
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(save_dir)

    with tf.compat.v1.Session() as sess:
      # Run the initializer on `w`.
      sess.run(init_op)
      output_val = sess.run(output, {input: [[1, 1], [1, 1]]})
      builder.add_meta_graph_and_variables(
          sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
          signature_def_map={
              "serving_default":
                  tf.compat.v1.saved_model \
                      .signature_def_utils.predict_signature_def(
                          inputs={"input": input},
                          outputs={"output": output})
          },
          assets_collection=None)

    builder.save()
    return {
        "async": False,
        "inputs": {
            "Placeholder": {
                "value": [[1, 1], [1, 1]], "shape": [2, 2], "dtype": 'float32'
            }
        },
        "outputs": {
            "MatMul": {
                "value": output_val.tolist(), "shape": [2, 2], "dtype": "float32"
            }
        }
    }

def _create_saved_model_v2(save_dir):
  """Test a basic TF V2 model with functions to make sure functions are inlined.

  Args:
    save_dir: directory name of where the saved model will be stored.
  """
  input_data = constant_op.constant(1., shape=[1])
  root = tracking.AutoTrackable()
  root.v1 = variables.Variable(3.)
  root.v2 = variables.Variable(2.)
  root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
  to_save = root.f.get_concrete_function(input_data)

  save(root, save_dir, to_save)
  return {
      "async": False,
      "inputs": {
          "x": {"value": [1], "shape": [1], "dtype": 'float32'}},
      "outputs": {
          "Identity:0": {"value": [6], "shape": [1], "dtype": "float32"}}}


def _create_saved_model_v2_with_control_flow(save_dir):
  """Test a basic TF v2 model with control flow to inlined.

  Args:
    save_dir: directory name of where the saved model will be stored.
  """
  @tf.function
  def square_if_positive(v):
    if v > 0:
      v = v * v
    else:
      v = v + 1
    return v

  root = tracking.AutoTrackable()
  root.f = square_if_positive
  to_save = root.f.get_concrete_function(
      tensor_spec.TensorSpec([], dtypes.float32))

  save(root, save_dir, to_save)
  print(square_if_positive(tf.constant(-2)))
  print(square_if_positive(tf.constant(3)))
  print(to_save.structured_input_signature)
  print(to_save.structured_outputs)
  return {
      "async": True,
      "inputs": {"v": {"value": 3, "shape": [], "dtype": 'float32'}},
      "outputs": {"Identity:0": {"value": [9], "shape": [], "dtype": "float32"}}}

def _create_saved_model_with_conv2d(save_dir):
  """Test a basic model with fusable conv2d.
  Args:
    save_dir: directory name of where the saved model will be stored.
  """
  layers = [
      tf.keras.layers.Conv2D(
          16, [3, 3], padding='same', use_bias=False),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.ReLU()
  ]
  model = tf.keras.Sequential(layers)
  result = model.predict(tf.ones((1, 24, 24, 3)))
  # set the learning phase to avoid keara learning placeholder, which
  # will cause error when saving.
  tf.keras.backend.set_learning_phase(0)
  tf.saved_model.save(model, save_dir)
  return {
      "async": False,
      "inputs": {
          "conv2d_input:0": {"value": np.ones((1, 24, 24, 3)).tolist(),
                "shape": [1, 24, 24, 3],
                "dtype": 'float32'}},
      "outputs": {
          "Identity:0": {"value": result.tolist(),
                         "shape": result.shape,
                         "dtype": "float32"}}}


def _create_saved_model_with_prelu(save_dir):
  """Test a basic model with prelu activation.
  Args:
    save_dir: directory name of where the saved model will be stored.
  """
  # set the bias and alpha intitialize to make them constant and ensure grappler
  # be able to fuse the op.
  layers = [
      tf.keras.layers.Conv2D(
          16, [3, 3], padding='same', use_bias=True,
          bias_initializer=tf.initializers.constant(0.25)),
      tf.keras.layers.PReLU(alpha_initializer=tf.initializers.constant(0.25))
  ]
  model = tf.keras.Sequential(layers)
  result = model.predict(tf.ones((1, 24, 24, 3)))
  tf.keras.backend.set_learning_phase(0)
  tf.saved_model.save(model, save_dir)
  return {
      "async": False,
      "inputs": {
          "conv2d_1_input": {"value": np.ones((1, 24, 24, 3)).tolist(),
                "shape": [1, 24, 24, 3],
                "dtype": 'float32'}},
      "outputs": {
          "Identity:0": {"value": result.tolist(),
                         "shape": result.shape,
                         "dtype": "float32"}}}

def _create_saved_model_v2_complex64(save_dir):
  """Test a TF V2 model with complex dtype.

  Args:
    save_dir: directory name of where the saved model will be stored.
  """
  input_data = constant_op.constant(1., shape=[1])
  root = tracking.AutoTrackable()
  root.v1 = variables.Variable(3 + 1j, dtype=tf.complex64)
  root.f = def_function.function(lambda x: tf.complex(x, x) + root.v1)
  to_save = root.f.get_concrete_function(input_data)

  save(root, save_dir, to_save)
  return {
      "async": False,
      "inputs": {
          "x": {"value": [1], "shape": [1], "dtype": 'float32'}},
      "outputs": {
          "Identity:0": {"value": [4, 2], "shape": [1], "dtype": "complex64"}}}

def _create_saved_model_v2_with_control_flow_v2(save_dir):
  """Test a TF V2 model with control flow v2.

  Args:
    save_dir: directory name of where the saved model will be stored.
  """
  class CustomModule(tf.Module):

    def __init__(self):
        super(CustomModule, self).__init__()

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32),
                                  tf.TensorSpec([], tf.int32),
                                  tf.TensorSpec([], tf.int32)])
    def control_flow(self, x, y, z):
        i = 0
        while i < z:
            i += 1
            j = 0
            while j < y:
                j += 1
                if z > 0:
                    x += 1
                else:
                    x += 2
        return x


  module = CustomModule()
  print(module.control_flow(0, 2, 10))
  tf.saved_model.save(module, save_dir,
                      signatures=module.control_flow)

  return {
      "async": False,
      "inputs": {
          "x": {"value": [0], "shape": [], "dtype": 'int32'},
          "y": {"value": [2], "shape": [], "dtype": 'int32'},
          "z": {"value": [10], "shape": [], "dtype": 'int32'}},
      "outputs": {
          "Identity:0": {"value": [20], "shape": [], "dtype": "int32"}}}

def _create_saved_model_v2_with_tensorlist_ops(save_dir):
  """Test a TF V2 model with TensorList Ops.

  Args:
    save_dir: directory name of where the saved model will be stored.
  """
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Embedding(100, 20, input_shape=[10]))
  model.add(tf.keras.layers.GRU(4))

  result = model.predict(tf.ones([1, 10]))

  tf.keras.backend.set_learning_phase(0)
  tf.saved_model.save(model, save_dir)

  return {
      "async": False,
      "inputs": {
          "embedding_input": {
            "value": np.ones((1, 10)).tolist(),
            "shape": [1, 10], "dtype": 'float32'}},
      "outputs": {
          "Identity:0": {
              "value": result.tolist(),
              "shape": result.shape,
              "dtype": "float32"}}}

def _create_saved_model_v1_with_hashtable(save_dir):
  """Test a TF V1 model with HashTable Ops.

  Args:
    save_dir: directory name of where the saved model will be stored.
  """
  graph = tf.Graph()

  with graph.as_default():
    # Create a builder.
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(save_dir)

    with tf.compat.v1.Session() as sess:
      keys_tensor = tf.constant(["a", "b"])
      vals_tensor = tf.constant([3, 4])

      table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys=keys_tensor, values=vals_tensor
        ),
        default_value=-1
      )
      input = tf.compat.v1.placeholder(tf.string, shape=[2])
      output = table.lookup(input)

      sess.run(tf.compat.v1.tables_initializer())

      # output_val = [3, -1]
      output_val = sess.run(output, {input: ["a", "c"]})

      builder.add_meta_graph_and_variables(
          sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
          signature_def_map={
              "serving_default":
                  tf.compat.v1.saved_model \
                      .signature_def_utils.predict_signature_def(
                          inputs={"input": input},
                          outputs={"output": output})
          },
          assets_collection=None)

    builder.save()
    return {
        "async": False,
        "inputs": {
            "Placeholder:0": {
                "value": ["a", "c"], "shape": [2], "dtype": "string"
            }
        },
        "outputs": {
            "hash_table_Lookup/LookupTableFindV2:0": {
                "value": output_val.tolist(), "shape": [2], "dtype": "int32"
            }
        }
    }

def _layers_mobilenet():
  model = tf.keras.applications.MobileNetV2()
  model_path = 'mobilenet'
  tfjs.converters.save_keras_model(model, os.path.join(
      _tmp_dir, model_path))
  xs_data_path = os.path.join(_tmp_dir, model_path + '.xs-data.json')
  xs_shape_path = os.path.join(_tmp_dir, model_path + '.xs-shapes.json')
  ys_data_path = os.path.join(_tmp_dir, model_path + '.ys-data.json')
  ys_shape_path = os.path.join(_tmp_dir, model_path + '.ys-shapes.json')

  input = tf.ones([1, 224, 224, 3])
  output = model.predict(input)

  with open(xs_data_path, 'w') as f:
    f.write(json.dumps([input.numpy().tolist()]))
  with open(xs_shape_path, 'w') as f:
    f.write(json.dumps([input.shape.as_list()]))
  with open(ys_data_path, 'w') as f:
    f.write(json.dumps([output.tolist()]))
  with open(ys_shape_path, 'w') as f:
    f.write(json.dumps([output.shape]))

def main():
  # Create the directory to store model and data.
  if os.path.exists(_tmp_dir) and os.path.isdir(_tmp_dir):
    shutil.rmtree(_tmp_dir)
  os.mkdir(_tmp_dir)

  _save_and_convert_model(_create_saved_model_v1, 'saved_model_v1')
  _save_and_convert_model(_create_saved_model_v2, 'saved_model_v2')
  _save_and_convert_model(_create_saved_model_v2_complex64,
                          'saved_model_v2_complex64')
  _save_and_convert_model(_create_saved_model_v2_with_control_flow,
      'saved_model_v2_with_control_flow')
  _save_and_convert_model(_create_saved_model_v2_with_control_flow_v2,
      'saved_model_v2_with_control_flow_v2', control_flow_v2=True)
  _save_and_convert_model(_create_saved_model_with_conv2d,
      'saved_model_with_conv2d')
  _save_and_convert_model(_create_saved_model_with_prelu,
      'saved_model_with_prelu')
  _save_and_convert_model(_create_saved_model_v2_with_tensorlist_ops,
      'saved_model_v2_with_tensorlist_ops', control_flow_v2=True)
  _save_and_convert_model(_create_saved_model_v1_with_hashtable,
      'saved_model_v1_with_hashtable')

  _layers_mobilenet()
if __name__ == '__main__':
  main()
