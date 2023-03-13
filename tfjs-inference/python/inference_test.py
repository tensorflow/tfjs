# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run `yarn test-python` in the package root directory.
# This test only supports running in Linux.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path
import tempfile
import shutil

import tensorflow as tf

import inference

class InferenceTest(tf.test.TestCase):

  def testInference(self):
    backends = ['cpu', 'wasm']
    for backend in backends:
      binary_path = os.path.join('../binaries', 'tfjs-inference-linux')
      model_path = os.path.join('../test_data', 'model.json')
      test_data_dir = os.path.join('../test_data')
      tmp_dir = tempfile.mkdtemp()

      inference.predict(binary_path, model_path, test_data_dir, tmp_dir, backend=backend)

      with open(os.path.join(tmp_dir, 'data.json'), 'rt') as f:
        ys_values = json.load(f)

        # The output is a list of tensor data in the form of dict.
        # Example output:
        # [{"0":0.7567615509033203,"1":-0.18349379301071167,"2":0.7567615509033203,"3":-0.18349379301071167}]
        ys_values = [list(y.values()) for y in ys_values]

      with open(os.path.join(tmp_dir, 'shape.json'), 'rt') as f:
        ys_shapes = json.load(f)

      with open(os.path.join(tmp_dir, 'dtype.json'), 'rt') as f:
        ys_dtypes = json.load(f)

      self.assertAllClose(ys_values[0], [
          0.7567615509033203, -0.18349379301071167, 0.7567615509033203,
          -0.18349379301071167
      ])
      self.assertAllEqual(ys_shapes[0], [2, 2])
      self.assertEqual(ys_dtypes[0], 'float32')
      self.assertFalse(os.path.exists(os.path.join(tmp_dir, 'name.json')))
      # Cleanup tmp dir.
      shutil.rmtree(tmp_dir)

  # Todo(linazhao): Add a test model that outputs multiple tensors.
  def testInferenceWithOutputNameFile(self):
    binary_path = os.path.join('../binaries', 'tfjs-inference-linux')
    model_path = os.path.join('../test_data', 'model.json')
    test_data_dir = os.path.join('../test_data')
    tmp_dir = tempfile.mkdtemp()

    inference.predict(binary_path, model_path, test_data_dir, tmp_dir, tf_output_name_file='tf_output_name.json')

    with open(os.path.join(tmp_dir, 'data.json'), 'rt') as f:
      ys_values = json.load(f)

      # The output is a list of tensor data in the form of dict.
      # Example output:
      # [{"0":0.7567615509033203,"1":-0.18349379301071167,"2":0.7567615509033203,"3":-0.18349379301071167}]
      ys_values = [list(y.values()) for y in ys_values]

    with open(os.path.join(tmp_dir, 'shape.json'), 'rt') as f:
      ys_shapes = json.load(f)

    with open(os.path.join(tmp_dir, 'dtype.json'), 'rt') as f:
      ys_dtypes = json.load(f)

    self.assertAllClose(ys_values[0], [
        0.7567615509033203, -0.18349379301071167, 0.7567615509033203,
        -0.18349379301071167
    ])
    self.assertAllEqual(ys_shapes[0], [2, 2])
    self.assertEqual(ys_dtypes[0], 'float32')
    self.assertFalse(os.path.exists(os.path.join(tmp_dir, 'name.json')))
    # Cleanup tmp dir.
    shutil.rmtree(tmp_dir)

  def testInferenceWithNonExistingOutputNameFile(self):
    binary_path = os.path.join('../binaries', 'tfjs-inference-linux')
    model_path = os.path.join('../test_data', 'model.json')
    test_data_dir = os.path.join('../test_data')
    tmp_dir = tempfile.mkdtemp()

    # Throws an error
    with self.assertRaises(ValueError):
      inference.predict(binary_path, model_path, test_data_dir, tmp_dir, tf_output_name_file='non_exist.json')

    # ...and does not create an output file.
    with self.assertRaises(FileNotFoundError):
      with open(os.path.join(tmp_dir, 'data.json'), 'rt') as f:
        json.load(f)

    # Cleanup tmp dir.
    shutil.rmtree(tmp_dir)

  def testInferenceWithStructuredOutputKeys(self):
    backends = ['cpu', 'wasm']
    for backend in backends:
      binary_path = os.path.join('../binaries', 'tfjs-inference-linux')
      model_path = os.path.join('../test_data', 'model_structured_outputs.json')
      test_data_dir = os.path.join('../test_data')
      tmp_dir = tempfile.mkdtemp()

      inference.predict(binary_path, model_path, test_data_dir, tmp_dir, backend=backend)

      with open(os.path.join(tmp_dir, 'data.json'), 'rt') as f:
        ys_values = json.load(f)

        # The output is a list of tensor data in the form of dict.
        # Example output:
        # [{"0":0.7567615509033203,"1":-0.18349379301071167,"2":0.7567615509033203,"3":-0.18349379301071167}]
        ys_values = [list(y.values()) for y in ys_values]

      with open(os.path.join(tmp_dir, 'shape.json'), 'rt') as f:
        ys_shapes = json.load(f)

      with open(os.path.join(tmp_dir, 'dtype.json'), 'rt') as f:
        ys_dtypes = json.load(f)

      with open(os.path.join(tmp_dir, 'name.json'), 'rt') as f:
        ys_names = json.load(f)

      self.assertAllClose(ys_values[0], [
          0.7567615509033203, -0.18349379301071167, 0.7567615509033203,
          -0.18349379301071167
      ])
      self.assertAllEqual(ys_shapes[0], [2, 2])
      self.assertEqual(ys_dtypes[0], 'float32')
      self.assertEqual(ys_names[0], 'testName')
      # Cleanup tmp dir.
      shutil.rmtree(tmp_dir)

if __name__ == '__main__':
  tf.test.main()
