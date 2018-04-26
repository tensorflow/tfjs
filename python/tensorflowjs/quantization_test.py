# Copyright 2018 Google LLC
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np

from tensorflowjs import quantization

class TestQuantizationUtil(unittest.TestCase):

  def _runQuantizeTest(
      self, range_min, range_max, data_dtype, quantization_dtype,
      expected_scale):
    d = np.arange(range_min, range_max + 1, dtype=data_dtype)
    q, s, m = quantization.quantize_weights(d, quantization_dtype)
    self.assertAlmostEqual(s, expected_scale)
    self.assertEqual(q.dtype, quantization_dtype)

    de_q = quantization.dequantize_weights(q, s, m, data_dtype)
    np.testing.assert_allclose(de_q, d)

    if range_min <= 0 <= range_max:
      d_0 = np.zeros(1, data_dtype)
      q_0 = np.round((d_0 - m) / s).astype(quantization_dtype)
      self.assertEqual(
          quantization.dequantize_weights(q_0, s, m, data_dtype), d_0)

  def testAllEqual(self):
    d = np.ones(5, dtype=np.float32)
    q, s, m = quantization.quantize_weights(d, np.uint8)
    self.assertEqual(s, 1.0)
    self.assertEqual(q.dtype, np.uint8)

    de_q = quantization.dequantize_weights(q, s, m, np.float32)
    np.testing.assert_array_equal(de_q, d)

  def testQuantizeNegativeFloats(self):
    self._runQuantizeTest(-3, -1, np.float32, np.uint8, expected_scale=2/255)
    self._runQuantizeTest(-3, -1, np.float32, np.uint16, expected_scale=2/65536)

  def testQuantizeNegativeAndZeroFloats(self):
    self._runQuantizeTest(-3, 0, np.float32, np.uint8, expected_scale=3/255)
    self._runQuantizeTest(-3, 0, np.float32, np.uint16, expected_scale=3/65536)

  def testQuantizeNegativeAndPositiveFloats(self):
    self._runQuantizeTest(-3, 3, np.float32, np.uint8, expected_scale=6/255)
    self._runQuantizeTest(-3, 3, np.float32, np.uint16, expected_scale=6/65536)

  def testQuantizeZeroAndPositiveFloats(self):
    self._runQuantizeTest(0, 3, np.float32, np.uint8, expected_scale=3/255)
    self._runQuantizeTest(0, 3, np.float32, np.uint16, expected_scale=3/65536)

  def testQuantizePositiveFloats(self):
    self._runQuantizeTest(1, 3, np.float32, np.uint8, expected_scale=2/255)
    self._runQuantizeTest(1, 3, np.float32, np.uint16, expected_scale=2/65536)

  def testQuantizeNegativeInts(self):
    self._runQuantizeTest(-3, -1, np.int32, np.uint8, expected_scale=2/255)
    self._runQuantizeTest(-3, -1, np.int32, np.uint16, expected_scale=2/65536)

  def testQuantizeNegativeAndZeroInts(self):
    self._runQuantizeTest(-3, 0, np.int32, np.uint8, expected_scale=3/255)
    self._runQuantizeTest(-3, 0, np.int32, np.uint16, expected_scale=3/65536)

  def testQuantizeNegativeAndPositiveInts(self):
    self._runQuantizeTest(-3, 3, np.int32, np.uint8, expected_scale=6/255)
    self._runQuantizeTest(-3, 3, np.int32, np.uint16, expected_scale=6/65536)

  def testQuantizeZeroAndPositiveInts(self):
    self._runQuantizeTest(0, 3, np.int32, np.uint8, expected_scale=3/255)
    self._runQuantizeTest(0, 3, np.int32, np.uint16, expected_scale=3/65536)

  def testQuantizePositiveInts(self):
    self._runQuantizeTest(1, 3, np.int32, np.uint8, expected_scale=2/255)
    self._runQuantizeTest(1, 3, np.int32, np.uint16, expected_scale=2/65536)


if __name__ == '__main__':
  unittest.main()
