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

  def assertDictContainsSubsetAlmostEqual(self, d1, d2):
    self.assertIsInstance(d1, dict)
    self.assertIsInstance(d2, dict)

    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())

    self.assertTrue(d2_keys.issubset(d1_keys))

    for key in d2_keys:
      self.assertAlmostEqual(d1[key], d2[key])


  def _runQuantizeTest(
      self, range_min, range_max, data_dtype, quantization_dtype,
      expected_metadata):
    d = np.arange(range_min, range_max + 1, dtype=data_dtype)
    q, metadata = quantization.quantize_weights(d, quantization_dtype)

    self.assertDictContainsSubsetAlmostEqual(metadata, expected_metadata)
    self.assertEqual(q.dtype, quantization_dtype)

    de_q = quantization.dequantize_weights(
        q, metadata, data_dtype)
    if data_dtype != np.float32:
      np.testing.assert_allclose(de_q, d)
    else:
      np.testing.assert_array_almost_equal(de_q, d, decimal=2)

    if quantization_dtype in [np.uint8, np.uint16]:
      s = metadata['scale']
      m = metadata['min']
      if range_min <= 0 <= range_max:
        d_0 = np.zeros(1, data_dtype)
        q_0 = np.round((d_0 - m) / s).astype(quantization_dtype)
        self.assertEqual(
            quantization.dequantize_weights(q_0, metadata, data_dtype), d_0)

  def testAffineQuantizeAllEqual(self):
    d = np.ones(5, dtype=np.float32)
    q, metadata = quantization.quantize_weights(d, np.uint8)
    assert 'scale' in metadata and 'min' in metadata
    self.assertEqual(metadata['scale'], 1.0)
    self.assertEqual(q.dtype, np.uint8)

    de_q = quantization.dequantize_weights(q, metadata, np.float32)
    np.testing.assert_array_equal(de_q, d)

  def testFloatQuantizeAllEqual(self):
    d = np.ones(5, dtype=np.float32)
    q, metadata = quantization.quantize_weights(d, np.float16)
    self.assertDictEqual(metadata, {})

    self.assertEqual(q.dtype, np.float16)
    de_q = quantization.dequantize_weights(q, metadata, np.float32)
    np.testing.assert_array_equal(de_q, d)

  def testAffineQuantizeNegativeFloats(self):
    self._runQuantizeTest(
        -3, -1, np.float32, np.uint8,
        expected_metadata={'scale': 2/255})
    self._runQuantizeTest(
        -3, -1, np.float32, np.uint16,
        expected_metadata={'scale': 2/65536})

  def testAffineQuantizeNegativeAndZeroFloats(self):
    self._runQuantizeTest(
        -3, 0, np.float32, np.uint8,
        expected_metadata={'scale': 3/255})
    self._runQuantizeTest(
        -3, 0, np.float32, np.uint16,
        expected_metadata={'scale': 3/65536})

  def testAffineQuantizeNegativeAndPositiveFloats(self):
    self._runQuantizeTest(
        -3, 3, np.float32, np.uint8,
        expected_metadata={'scale': 6/255})
    self._runQuantizeTest(
        -3, 3, np.float32, np.uint16,
        expected_metadata={'scale': 6/65536})

  def testAffineQuantizeZeroAndPositiveFloats(self):
    self._runQuantizeTest(
        0, 3, np.float32, np.uint8,
        expected_metadata={'scale': 3/255})
    self._runQuantizeTest(
        0, 3, np.float32, np.uint16,
        expected_metadata={'scale': 3/65536})

  def testAffineQuantizePositiveFloats(self):
    self._runQuantizeTest(
        1, 3, np.float32, np.uint8,
        expected_metadata={'scale': 2/255})
    self._runQuantizeTest(
        1, 3, np.float32, np.uint16,
        expected_metadata={'scale': 2/65536})

  def testAffineQuantizeNormalizedFloats(self):
    data = np.array(
        [-0.29098126, -0.24776903, -0.27248842, 0.23848203], dtype=np.float32)
    q, metadata = quantization.quantize_weights(data, np.uint16)
    de_q = quantization.dequantize_weights(q, metadata, data.dtype)
    np.testing.assert_array_almost_equal(de_q, data, decimal=5)

  def testAffineQuantizeNegativeInts(self):
    self._runQuantizeTest(
        -3, -1, np.int32, np.uint8,
        expected_metadata={'scale': 2/255})
    self._runQuantizeTest(
        -3, -1, np.int32, np.uint16,
        expected_metadata={'scale': 2/65536})

  def testAffineQuantizeNegativeAndZeroInts(self):
    self._runQuantizeTest(
        -3, 0, np.int32, np.uint8,
        expected_metadata={'scale': 3/255})
    self._runQuantizeTest(
        -3, 0, np.int32, np.uint16,
        expected_metadata={'scale': 3/65536})

  def testAffineQuantizeNegativeAndPositiveInts(self):
    self._runQuantizeTest(
        -3, 3, np.int32, np.uint8,
        expected_metadata={'scale': 6/255})
    self._runQuantizeTest(
        -3, 3, np.int32, np.uint16,
        expected_metadata={'scale': 6/65536})

  def testAffineQuantizeZeroAndPositiveInts(self):
    self._runQuantizeTest(
        0, 3, np.int32, np.uint8,
        expected_metadata={'scale': 3/255})
    self._runQuantizeTest(
        0, 3, np.int32, np.uint16,
        expected_metadata={'scale': 3/65536})

  def testAffineQuantizePositiveInts(self):
    self._runQuantizeTest(
        1, 3, np.int32, np.uint8,
        expected_metadata={'scale': 2/255})
    self._runQuantizeTest(
        1, 3, np.int32, np.uint16,
        expected_metadata={'scale': 2/65536})

  def testInvalidQuantizationTypes(self):
    # Invalid quantization type
    with self.assertRaises(ValueError):
      quantization.quantize_weights(np.array([]), np.bool)

    # Invalid data dtype for float16 quantization
    with self.assertRaises(ValueError):
      d = np.ones(1, dtype=np.int32)
      quantization.quantize_weights(d, np.float16)

  def testInvalidDequantizationTypes(self):
    # Invalid metadata for affine quantization
    with self.assertRaises(ValueError):
      d = np.ones(1, dtype=np.uint8)
      quantization.dequantize_weights(np.array([]), {})

    # Invalid target dtype for float16 quantization
    with self.assertRaises(ValueError):
      d = np.ones(1, dtype=np.float16)
      quantization.dequantize_weights(d, {}, np.int32)

    # Invalid dequantization type
    with self.assertRaises(ValueError):
      d = np.ones(1, dtype=np.bool)
      quantization.dequantize_weights(d, {})

  def testMapLayerFallthrough(self):
    names = ['conv/0/weight', 'conv/0/bias', 'conv/1/weight', 'conv/1/bias']
    quantization_dtype_map = {'float16': ['conv/0/*'], 'uint8': True}
    dtype_map = quantization.map_layers_to_quantization_dtype(
        names, quantization_dtype_map)

    self.assertDictEqual(dtype_map, {
        'conv/0/weight': np.float16,
        'conv/0/bias': np.float16,
        'conv/1/weight': np.uint8,
        'conv/1/bias': np.uint8
    })

  def testMapLayerConflictingMap(self):
    names = ['conv/0/weight', 'conv/0/bias', 'conv/1/weight', 'conv/1/bias']
    quantization_dtype_map = {'float16': ['conv/0/*'], 'uint8': ['conv/0/bias']}

    with self.assertRaises(ValueError):
      quantization.map_layers_to_quantization_dtype(
          names, quantization_dtype_map)


  def testMapLayerStringToList(self):
    names = ['conv/0/weight', 'conv/0/bias', 'conv/1/weight', 'conv/1/bias']
    quantization_dtype_map = {'float16': '*'}


    dtype_map = quantization.map_layers_to_quantization_dtype(
        names, quantization_dtype_map)

    self.assertDictEqual(dtype_map, {
        'conv/0/weight': np.float16,
        'conv/0/bias': np.float16,
        'conv/1/weight': np.float16,
        'conv/1/bias': np.float16
    })

  def testMapLayerNoDtypeMap(self):
    names = ['conv/0/weight', 'conv/0/bias', 'conv/1/weight', 'conv/1/bias']
    quantization_dtype_map = {}
    dtype_map = quantization.map_layers_to_quantization_dtype(
        names, quantization_dtype_map)

    self.assertDictEqual(dtype_map, {})

  def testMapLayerExactlyOneFallthrough(self):
    names = ['conv/0/weight', 'conv/0/bias', 'conv/1/weight', 'conv/1/bias']
    quantization_dtype_map = {'float16': True, 'uint8': True}

    with self.assertRaises(ValueError):
      quantization.map_layers_to_quantization_dtype(
          names, quantization_dtype_map)



if __name__ == '__main__':
  unittest.main()
