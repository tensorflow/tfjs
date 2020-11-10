# -*- coding: utf-8 -*-
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

import os
import shutil

import tempfile

import numpy as np
import tensorflow as tf

from tensorflowjs import read_weights
from tensorflowjs import write_weights


class ReadWeightsTest(tf.test.TestCase):
  def setUp(self):
    self._tmp_dir = tempfile.mkdtemp()
    super(ReadWeightsTest, self).setUp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)
    super(ReadWeightsTest, self).tearDown()

  def testReadOneGroup(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float32')
        }, {
            'name': 'weight2',
            'data': np.array([1 + 1j, 2 + 2j, 3 + 3j])
        }]
    ]

    manifest = write_weights.write_weights(groups, self._tmp_dir)

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(manifest, self._tmp_dir)
    self.assertEqual(1, len(read_output))
    self.assertEqual(2, len(read_output[0]))
    self.assertEqual('weight1', read_output[0][0]['name'])
    self.assertTrue(
        np.allclose(groups[0][0]['data'], read_output[0][0]['data']))
    self.assertEqual('weight2', read_output[0][1]['name'])
    self.assertTrue(
        np.allclose(groups[0][1]['data'], read_output[0][1]['data']))
  def testReadOneGroupString(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([['test', 'a'], ['b', 'c']], 'object')
        }]
    ]

    manifest = write_weights.write_weights(groups, self._tmp_dir)

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(manifest, self._tmp_dir)
    self.assertEqual(1, len(read_output))
    self.assertEqual(1, len(read_output[0]))
    self.assertEqual('weight1', read_output[0][0]['name'])
    np.testing.assert_array_equal(
        read_output[0][0]['data'],
        np.array([[u'test'.encode('utf-8'), u'a'.encode('utf-8')],
                  [u'b'.encode('utf-8'), u'c'.encode('utf-8')]], 'object'))


  def testReadCyrillicStringUnicodeAndEncoded(self):
    groups = [
        [{
            'name': 'weight1',
            # String is stored as unicode.
            'data': np.array([u'здраво'], 'object')
        }, {
            'name': 'weight2',
            # String is stored encoded.
            'data': np.array([u'поздрав'.encode('utf-8')], 'object')
        }, {
            'name': 'weight3',
            # Let np choose the dtype (string or bytes depending on py version).
            'data': np.array([u'здраво'.encode('utf-8')])
        }]
    ]

    manifest = write_weights.write_weights(groups, self._tmp_dir)

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(manifest, self._tmp_dir)
    self.assertEqual(1, len(read_output))
    group = read_output[0]
    self.assertEqual(3, len(group))

    weight1 = group[0]
    self.assertEqual('weight1', weight1['name'])
    np.testing.assert_array_equal(
        weight1['data'],
        np.array([u'здраво'.encode('utf-8')], 'object'))

    weight2 = group[1]
    self.assertEqual('weight2', weight2['name'])
    np.testing.assert_array_equal(
        weight2['data'],
        np.array([u'поздрав'.encode('utf-8')], 'object'))

    weight3 = group[2]
    self.assertEqual('weight3', weight3['name'])
    np.testing.assert_array_equal(
        weight3['data'],
        np.array([u'здраво'.encode('utf-8')]))

  def testReadEastAsianStringUnicodeAndEncoded(self):
    # Each string tensor uses different encoding.
    groups = [
        [{
            'name': 'weight1',
            # Decoded.
            'data': np.array([u'语言处理'], 'object')
        }, {
            'name': 'weight2',
            # Encoded as utf-16.
            'data': np.array([u'语言处理'.encode('utf-16')], 'object')
        }, {
            'name': 'weight3',
            # Encoded as utf-8.
            'data': np.array([u'语言处理'.encode('utf-8')], 'object')
        }]
    ]

    manifest = write_weights.write_weights(groups, self._tmp_dir)

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(manifest, self._tmp_dir)
    self.assertEqual(1, len(read_output))
    group = read_output[0]
    self.assertEqual(3, len(group))

    weight1 = group[0]
    self.assertEqual('weight1', weight1['name'])
    np.testing.assert_array_equal(
        weight1['data'],
        np.array([u'语言处理'.encode('utf-8')], 'object'))

    weight2 = group[1]
    self.assertEqual('weight2', weight2['name'])
    np.testing.assert_array_equal(
        weight2['data'],
        np.array([u'语言处理'.encode('utf-16')], 'object'))

    weight3 = group[2]
    self.assertEqual('weight3', weight3['name'])
    np.testing.assert_array_equal(
        weight3['data'],
        np.array([u'语言处理'.encode('utf-8')], 'object'))

  def testReadOneGroupStringWithShards(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array(['test', 'a', 'c'], 'object')
        }]
    ]

    manifest = write_weights.write_weights(groups, self._tmp_dir,
                                           shard_size_bytes=4)

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(manifest, self._tmp_dir)
    self.assertEqual(1, len(read_output))
    self.assertEqual(1, len(read_output[0]))
    self.assertEqual('weight1', read_output[0][0]['name'])
    np.testing.assert_array_equal(read_output[0][0]['data'],
                                  np.array([u'test'.encode('utf-8'),
                                            u'a'.encode('utf-8'),
                                            u'c'.encode('utf-8')], 'object'))

  def testReadOneGroupEmptyStrings(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array(['', ''], 'object')
        }, {
            'name': 'weight2',
            'data': np.array([], 'object')
        }, {
            'name': 'weight3',
            'data': np.array([[]], 'object')
        }]
    ]

    manifest = write_weights.write_weights(groups, self._tmp_dir)

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(manifest, self._tmp_dir)
    self.assertEqual(1, len(read_output))
    group = read_output[0]
    self.assertEqual(3, len(group))

    weight1 = group[0]
    self.assertEqual('weight1', weight1['name'])
    np.testing.assert_array_equal(
        weight1['data'],
        np.array([u''.encode('utf-8'), u''.encode('utf-8')], 'object'))

    weight2 = group[1]
    self.assertEqual('weight2', weight2['name'])
    np.testing.assert_array_equal(
        weight2['data'],
        np.array([], 'object'))

    weight3 = group[2]
    self.assertEqual('weight3', weight3['name'])
    np.testing.assert_array_equal(
        weight3['data'],
        np.array([[]], 'object'))

  def testReadOneGroupFlattened(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float32')
        }]
    ]

    manifest = write_weights.write_weights(groups, self._tmp_dir)

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(
        manifest, self._tmp_dir, flatten=True)
    self.assertEqual(1, len(read_output))
    self.assertEqual('weight1', read_output[0]['name'])
    self.assertTrue(np.allclose(groups[0][0]['data'], read_output[0]['data']))

  def testReadOneGroupWithInt32DataFlattened(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float32')
        }, {
            'name': 'weight2',
            'data': np.array([10, 20, 30], 'int32')
        }]
    ]

    manifest = write_weights.write_weights(groups, self._tmp_dir)

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(
        manifest, self._tmp_dir, flatten=True)
    self.assertEqual(2, len(read_output))
    self.assertEqual('weight1', read_output[0]['name'])
    self.assertTrue(np.allclose(groups[0][0]['data'], read_output[0]['data']))
    self.assertEqual('weight2', read_output[1]['name'])
    self.assertTrue(np.allclose(groups[0][1]['data'], read_output[1]['data']))

  def testReadTwoGroupsFlattend(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float32')
        }],
        [{
            'name': 'weight2',
            'data': np.array([10, 20, 30], 'int32')
        }]
    ]

    manifest = write_weights.write_weights(groups, self._tmp_dir)

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(
        manifest, self._tmp_dir, flatten=True)
    self.assertEqual(2, len(read_output))
    self.assertEqual('weight1', read_output[0]['name'])
    self.assertTrue(np.allclose(groups[0][0]['data'], read_output[0]['data']))
    self.assertEqual('weight2', read_output[1]['name'])
    self.assertTrue(np.allclose(groups[1][0]['data'], read_output[1]['data']))

  def testReadOneGroupWithShards(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.random.rand(1, 100).astype(np.float32)
        }]
    ]

    manifest = write_weights.write_weights(groups, self._tmp_dir)

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(manifest, self._tmp_dir)
    self.assertEqual(1, len(read_output))
    self.assertEqual(1, len(read_output[0]))
    self.assertEqual('weight1', read_output[0][0]['name'])
    self.assertTrue(
        np.allclose(groups[0][0]['data'], read_output[0][0]['data']))

  def testReadWeightsWithIncorrectTypeInWeightsManifestRaisesError(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.random.rand(1, 100).astype(np.float32)
        }]
    ]

    write_weights.write_weights(groups, self._tmp_dir)

    with self.assertRaises(ValueError):
      read_weights.read_weights(groups[0][0], self._tmp_dir)


  def testReadAffineQuantizedWeights(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([0, 1, 2, 3], 'float32')
        }]
    ]

    manifest = write_weights.write_weights(
        groups, self._tmp_dir, quantization_dtype_map={'uint8': '*'})

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(manifest, self._tmp_dir)
    self.assertEqual(1, len(read_output))
    self.assertEqual(1, len(read_output[0]))
    self.assertEqual('weight1', read_output[0][0]['name'])
    self.assertEqual(read_output[0][0]['data'].dtype, np.float32)
    self.assertTrue(
        np.allclose(groups[0][0]['data'], read_output[0][0]['data']))

  def testReadFloat16QuantizedWeights(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([0, 1, 2, 3], 'float32')
        }]
    ]

    manifest = write_weights.write_weights(
        groups, self._tmp_dir, quantization_dtype_map={'float16': '*'})

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(manifest, self._tmp_dir)
    self.assertEqual(1, len(read_output))
    self.assertEqual(1, len(read_output[0]))
    self.assertEqual('weight1', read_output[0][0]['name'])
    self.assertEqual(read_output[0][0]['data'].dtype, np.float32)
    self.assertTrue(
        np.allclose(groups[0][0]['data'], read_output[0][0]['data']))

  def testReadBoolWeights(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([True, False, True], 'bool')
        }]
    ]

    manifest = write_weights.write_weights(groups, self._tmp_dir)

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(manifest, self._tmp_dir)
    self.assertEqual(1, len(read_output))
    self.assertEqual(1, len(read_output[0]))
    self.assertEqual('weight1', read_output[0][0]['name'])
    np.testing.assert_array_equal(read_output[0][0]['data'],
                                  np.array([True, False, True], 'bool'))

  def testReadStringScalar(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array(u'abc'.encode('utf-8'), 'object')
        }]
    ]

    manifest = write_weights.write_weights(groups, self._tmp_dir)

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(manifest, self._tmp_dir)
    self.assertEqual(1, len(read_output))
    self.assertEqual(1, len(read_output[0]))
    self.assertEqual('weight1', read_output[0][0]['name'])
    np.testing.assert_array_equal(read_output[0][0]['data'],
                                  np.array(u'abc'.encode('utf-8'), 'object'))

if __name__ == '__main__':
  tf.test.main()
