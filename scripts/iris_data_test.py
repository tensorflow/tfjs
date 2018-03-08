# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Test for the Iris dataset module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np

from scripts import iris_data


class IrisDataTest(unittest.TestCase):

  def testLoadData(self):
    iris_x, iris_y = iris_data.load()
    self.assertEqual(2, len(iris_x.shape))
    self.assertGreater(iris_x.shape[0], 0)
    self.assertEqual(4, iris_x.shape[1])
    self.assertEqual(iris_x.shape[0], iris_y.shape[0])
    self.assertEqual(3, iris_y.shape[1])
    self.assertTrue(
        np.allclose(np.ones([iris_y.shape[0], 1]), np.sum(iris_y, axis=1)))


if __name__ == '__main__':
  unittest.main()
