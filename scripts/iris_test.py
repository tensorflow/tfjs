# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Test for the Iris model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
import tempfile
import unittest

from scripts import iris


class IrisTest(unittest.TestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()
    super(IrisTest, self).setUp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(IrisTest, self).tearDown()

  def testTrainAndSaveNonSequential(self):
    model_json_path = os.path.join(self._tmp_dir, 'iris.keras.model.json')
    weights_json_path = os.path.join(self._tmp_dir, 'iris.keras.weights.json')
    merged_json_path = os.path.join(self._tmp_dir, 'iris.keras.merged.json')
    final_train_accuracy = iris.train(
        100, model_json_path, weights_json_path, merged_json_path,
        sequential=False)
    self.assertGreater(final_train_accuracy, 0.9)

    # Check that the model json file is created.
    json.load(open(model_json_path, 'rt'))

    # Check that the weights json file is created.
    json.load(open(weights_json_path, 'rt'))

    # Check that the merged json file is created.
    json.load(open(merged_json_path, 'rt'))

  def testTrainAndSaveSequential(self):
    model_json_path = os.path.join(self._tmp_dir, 'iris.keras.model.json')
    weights_json_path = os.path.join(self._tmp_dir, 'iris.keras.weights.json')
    merged_json_path = os.path.join(self._tmp_dir, 'iris.keras.merged.json')
    final_train_accuracy = iris.train(
        100, model_json_path, weights_json_path, merged_json_path,
        sequential=True)
    self.assertGreater(final_train_accuracy, 0.9)

    # Check that the model json file is created.
    json.load(open(model_json_path, 'rt'))

    # Check that the weights json file is created.
    json.load(open(weights_json_path, 'rt'))

    # Check that the merged json file is created.
    json.load(open(merged_json_path, 'rt'))


if __name__ == '__main__':
  unittest.main()
