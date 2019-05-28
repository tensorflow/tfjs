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
  """Create a TensorFlow SavedModel for testing."""

  graph = tf.Graph()
  with graph.as_default():
    input = tf.compat.v1.placeholder(tf.float32, shape=[2, 2])
    x = tf.compat.v1.constant([[37.0, -23.0], [1.0, 4.0]])
    w = tf.compat.v1.get_variable('w', shape=[2, 2])
    output = tf.compat.v1.matmul(tf.compat.v1.matmul(input, x), w)
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
    return {"async": False, "inputs": {"Placeholder": {"value": [[1, 1], [1, 1]], "shape": [2, 2], "dtype": 'float32'}},
            "outputs": {"MatMul_1": {"value": output_val.tolist(), "shape": [2,2], "dtype": "float32"}}}

def _create_saved_model(save_dir):
  """Test a basic model with functions to make sure functions are inlined."""
  input_data = constant_op.constant(1., shape=[1])
  root = tracking.AutoTrackable()
  root.v1 = variables.Variable(3.)
  root.v2 = variables.Variable(2.)
  root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
  to_save = root.f.get_concrete_function(input_data)

  save(root, save_dir, to_save)
  print(to_save.structured_input_signature)
  print(to_save.structured_outputs)
  return {"async": False, "inputs": {"x": {"value": [1], "shape": [1], "dtype": 'float32'}},
          "outputs": {"Identity:0": {"value": [6], "shape": [1], "dtype": "float32"}}}

def _create_saved_model_with_control_flow(save_dir):
  """Test a basic model with control flow to inlined."""
  @tf.function
  def find_next_odd(v):
    v1 = v + 1
    while tf.equal(v1 % 2, 0):
      v1 = v1 + 1
    return v1
  root = tracking.AutoTrackable()
  root.f = find_next_odd
  to_save = root.f.get_concrete_function(
      tensor_spec.TensorSpec([], dtypes.int32))

  save(root, save_dir, to_save)
  print(to_save.structured_input_signature)
  print(to_save.structured_outputs)
  return {"async": True, "inputs": {"v": {"value": 3, "shape": [], "dtype": 'int32'}},
          "outputs": {"Identity:0": {"value": [5], "shape": [], "dtype": "int32"}}}

def save_and_convert_model(model_name,
                           description,
                           model_fn,
                           artifacts_dir):
  """Benchmark a model's fit() and predict() calls; serialize the model.

  Args:
    model_fn: A function that creates the saved model.
    artifacts_dir: Directory to save the data in. The data includes:
      * topology and weights of the models, in TensorFlow.js format
      * metadata and benchmark information in a file named `data.json`,
        including:
        - the name and description of the model
        - the input and output shapes of the model
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
       _create_saved_model,
       'Saved model v2'),
      ('saved_model_v2_control_flow',
       _create_saved_model_with_control_flow,
       'Saved model v2 with control flow')
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
