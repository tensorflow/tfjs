# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example showing how to use tfjs binary in python code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess


def predict(binary_path,
            model_path,
            inputs_dir,
            outputs_dir,
            backend=None,
            tf_output_name_file=None):
  """Use tfjs binary to make inference and store output in file.

  Args:
    binary_path: Path to the nodejs binary. The path can be an absolute path
      (preferred) or a relative path from this python script's current
      directory.
    model_path: Directory to TensorFlow.js model's json file.
    inputs_dir: Directory to the inputs files, including data, shape and dtype
      files.
    outputs_dir: Directory to write the outputs files, including data, shape
      and dtype files.
    backend: Optional. Choose which TensorFlow.js backend to use. Supported
      backends include cpu and wasm. Default: cpu
    tf_output_name_file: Optional. File name of the tf_output_name, if file does
      not exist, will use the default outputs of the model.
  """
  model_path_option = '--model_path=' + model_path
  inputs_dir_option = '--inputs_dir=' + inputs_dir
  outputs_dir_option = '--outputs_dir=' + outputs_dir

  tfjs_inference_command = [
      binary_path, model_path_option, inputs_dir_option,
      outputs_dir_option
  ]

  if tf_output_name_file:
    tf_output_name_file_option = '--tf_output_name_file=' + tf_output_name_file
    tfjs_inference_command.append(tf_output_name_file_option)

  if backend:
    backend_option = '--backend=' + backend
    tfjs_inference_command.append(backend_option)

  popen = subprocess.Popen(
      tfjs_inference_command,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE)
  stdout, stderr = popen.communicate()

  if popen.returncode != 0:
    raise ValueError('Inference failed with status %d\nstderr:\n%s' %
                     (popen.returncode, stderr))
