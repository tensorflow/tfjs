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

import json
import os
import shutil
import unittest

import tempfile

import numpy as np

from tensorflowjs import read_weights
from tensorflowjs import write_weights


class ReadWeightsTest(unittest.TestCase):
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
        }]
    ]

    manifest_json = write_weights.write_weights(groups, self._tmp_dir)
    manifest = json.loads(manifest_json)

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(manifest, self._tmp_dir)
    self.assertEqual(1, len(read_output))
    self.assertEqual(1, len(read_output[0]))
    self.assertEqual('weight1', read_output[0][0]['name'])
    self.assertTrue(
        np.allclose(groups[0][0]['data'], read_output[0][0]['data']))

  def testReadOneGroupFlattened(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float32')
        }]
    ]

    manifest_json = write_weights.write_weights(groups, self._tmp_dir)
    manifest = json.loads(manifest_json)

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

    manifest_json = write_weights.write_weights(groups, self._tmp_dir)
    manifest = json.loads(manifest_json)

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

    manifest_json = write_weights.write_weights(groups, self._tmp_dir)
    manifest = json.loads(manifest_json)

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

    manifest_json = write_weights.write_weights(groups, self._tmp_dir)
    manifest = json.loads(manifest_json)

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


  def testReadQuantizedWeights(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([0, 1, 2, 3], 'float32')
        }]
    ]

    manifest_json = write_weights.write_weights(
        groups, self._tmp_dir, quantization_dtype=np.uint8)
    manifest = json.loads(manifest_json)

    # Read the weights using `read_weights`.
    read_output = read_weights.read_weights(manifest, self._tmp_dir)
    self.assertEqual(1, len(read_output))
    self.assertEqual(1, len(read_output[0]))
    self.assertEqual('weight1', read_output[0][0]['name'])
    self.assertTrue(
        np.allclose(groups[0][0]['data'], read_output[0][0]['data']))


if __name__ == '__main__':
  unittest.main()
