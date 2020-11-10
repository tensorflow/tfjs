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

"""Validate outputs of TensorFlow.js tfjs-converter graph models.
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

_PREDICT_BURNINS = 1  # How many predict() runs to do before timing predict().
_PREDICT_RUNS = 1  # How many runs of predict() to average over.


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
          "input_1": {"value": np.ones((1, 24, 24, 3)).tolist(),
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
          "input_1": {"value": np.ones((1, 24, 24, 3)).tolist(),
                "shape": [1, 24, 24, 3],
                "dtype": 'float32'}},
      "outputs": {
          "Identity:0": {"value": result.tolist(),
                         "shape": result.shape,
                         "dtype": "float32"}}}

def save_and_convert_model(model_name,
                           description,
                           model_fn,
                           artifacts_dir):
  """Benchmark a model's fit() and predict() calls; serialize the model.

  Args:
    model_name: Name string for the model.
    description: Description string for the model.
    model_fn: A function that creates the saved model.
    artifacts_dir: Directory to save the data in. The data includes:
      * topology and weights of the models, in TensorFlow.js format
      * metadata and benchmark information in a file named `data.json`,
        including:
        - the name and description of the model
        - the input and output shapes of the model

  Returns:
    predict task_log hash that specifies the inputs and outputs for
    validation test.
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

  tmp_saved_model_dir = tempfile.mkdtemp()
  tasks = model_fn(tmp_saved_model_dir)

  subprocess.check_output([
      'tensorflowjs_converter',
      '--input_format', 'tf_saved_model',
      '--output_format', 'tfjs_graph_model',
      '--signature_name', 'serving_default',
      '--saved_model_tags', 'serve',
      tmp_saved_model_dir, artifacts_dir])
  # Clean up the temporary SavedModel directory.
  shutil.rmtree(tmp_saved_model_dir)

  # Collect and format the data for predict().
  task_logs['predict'] = {  # For schema, see 'ModelTaskLog` in types.ts.
    'taskType': 'model',
    'modelFormat': 'GraphModel',
    'modelName': model_name,
    'modelDescription': description,
    'functionName': 'predict',
    'inputs': tasks["inputs"],
    'outputs': tasks["outputs"],
    'async': tasks["async"]
  }
  return task_logs


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
  return environment_info


def main():
  environment_info = _get_python_environment_info()
  print('Environment info:')
  print(json.dumps(environment_info, indent=2))

  suite_log = dict()  # For schema, see `SuiteLog` in types.ts.
  suite_log['data'] = {}
  suite_log['environmentInfo'] = environment_info

  names_fns_and_descriptions = [
      ('saved_model_v1',
       _create_saved_model_v1,
       'Saved model v1'),
      ('saved_model_v2',
       _create_saved_model_v2,
       'Saved model v2'),
      ('saved_model_v2_control_flow',
       _create_saved_model_v2_with_control_flow,
       'Saved model v2 with control flow'),
      ('saved_model_v2_conv2d',
       _create_saved_model_with_conv2d,
       'Saved model v2 with conv2d'),
      ('saved_model_v2_prelu',
       _create_saved_model_with_prelu,
       'Saved model v2 with prelu activation')
       ]

  for model_name, model_fn, description in names_fns_and_descriptions:
    suite_log['data'][model_name] = (
        save_and_convert_model(
            model_name,
            description,
            model_fn,
            os.path.join(FLAGS.data_root, model_name)))

  with open(os.path.join(FLAGS.data_root, 'validations.json'), 'wt') as f:
    json.dump(suite_log, f)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Benchmarks demo.')
  parser.add_argument(
      'data_root',
      type=str,
      help='Local path for saving the results of benchmarks.')

  FLAGS, _ = parser.parse_known_args()
  main()
