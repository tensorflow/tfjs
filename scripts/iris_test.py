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
    final_train_accuracy = iris.train(100, self._tmp_dir)
    self.assertGreater(final_train_accuracy, 0.9)

    # Check that the model json file is created.
    json.load(open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))

  def testTrainAndSaveSequential(self):
    final_train_accuracy = iris.train(100, self._tmp_dir, sequential=True)
    self.assertGreater(final_train_accuracy, 0.9)

    # Check that the model json file is created.
    json.load(open(os.path.join(self._tmp_dir, 'model.json'), 'rt'))


if __name__ == '__main__':
  unittest.main()
