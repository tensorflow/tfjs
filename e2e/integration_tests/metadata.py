# @license
# Copyright 2020 Google LLC. All Rights Reserved.
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
_tmp_dir = os.path.join(curr_dir, 'metadata')

def _create_model_with_metadata():
  # Generate model, inputs, and outputs using Tensorflow.
  tmp_saved_model_dir = tempfile.mkdtemp()
  model_info = _create_saved_model(tmp_saved_model_dir)

  metadata1 = {'a': 1}
  metadata2 = {'label1': 0, 'label2': 1}
  metadata1_path = os.path.join(_tmp_dir, 'metadata1.json')
  metadata2_path = os.path.join(_tmp_dir, 'metadata2.json')
  with open(metadata1_path, 'w') as f:
    f.write(json.dumps(metadata1))
  with open(metadata2_path, 'w') as f:
    f.write(json.dumps(metadata2))
  metadata_option = 'metadata1:'+metadata1_path+','+'metadata2:'+metadata2_path

  # Convert and store model to file.
  args = [
      'tensorflowjs_converter',
      '--input_format', 'tf_saved_model',
      '--output_format', 'tfjs_graph_model',
      '--signature_name', 'serving_default',
      '--saved_model_tags', 'serve',
      '--metadata', metadata_option];

  print(args, tmp_saved_model_dir, _tmp_dir)
  subprocess.check_output(args +[tmp_saved_model_dir, _tmp_dir])

def _create_saved_model(save_dir):
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

def main():
  # Create the directory to store model and data.
  if os.path.exists(_tmp_dir) and os.path.isdir(_tmp_dir):
    shutil.rmtree(_tmp_dir)
  os.mkdir(_tmp_dir)

  _create_model_with_metadata()

if __name__ == '__main__':
  main()
