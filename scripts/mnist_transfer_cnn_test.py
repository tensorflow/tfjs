# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Test for the MNIST transfer learning CNN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from scripts import mnist_transfer_cnn


class MnistTest(unittest.TestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()
    super(MnistTest, self).setUp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(MnistTest, self).tearDown()

  def _getFakeMnistData(self):
    """Generate some fake MNIST data for testing."""
    x_train = np.random.rand(4, 28, 28)
    y_train = np.random.rand(4,)
    x_test = np.random.rand(4, 28, 28)
    y_test = np.random.rand(4,)
    return x_train, y_train, x_test, y_test

  def testWriteGte5DataToJsFiles(self):
    x_train, y_train, x_test, y_test = self._getFakeMnistData()
    js_file_prefix = os.path.join(self._tmp_dir, 'gte5')
    mnist_transfer_cnn.write_gte5_data(x_train, y_train, x_test, y_test,
                                       js_file_prefix)

    for js_path_suffix in ('.train.js', '.test.js'):
      with open(js_file_prefix + js_path_suffix, 'rt') as f:
        js_string = f.read().strip()
        # Remove the leading 'const ...' and trailing ';'.
        json_string = js_string[js_string.index('['):][:-1]
        data = json.loads(json_string)
        if 'train' in js_path_suffix:
          self.assertEqual(x_train.shape[0], len(data))
        else:
          self.assertEqual(x_test.shape[0], len(data))
        self.assertEqual(28, len(data[0]['x']))
        self.assertEqual(28, len(data[0]['x'][0]))

  def testTrainWithFakeDataAndSave(self):
    model_json_path = os.path.join(self._tmp_dir, 'mnist.keras.model.json')
    weights_json_path = os.path.join(self._tmp_dir, 'mnist.keras.weights.json')
    merged_json_path = os.path.join(self._tmp_dir, 'mnist.keras.merged.json')
    x_train, y_train, x_test, y_test = self._getFakeMnistData()
    mnist_transfer_cnn.train_and_save_model(
        2, 2, 2, 2, 1,
        x_train, y_train, x_test, y_test,
        model_json_path, weights_json_path, merged_json_path,
        optimizer='adam')

    # Check that the model json file is created.
    json.load(open(model_json_path, 'rt'))

    # Check that the weights json file is created.
    json.load(open(weights_json_path, 'rt'))

    # Check that the merged json file is created.
    json.load(open(merged_json_path, 'rt'))


if __name__ == '__main__':
  unittest.main()
