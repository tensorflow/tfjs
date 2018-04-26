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

import json
import os
import shutil
import unittest

import numpy as np

from tensorflowjs import quantization
from tensorflowjs import write_weights

TMP_DIR = '/tmp/write_weights_test/'


class TestWriteWeights(unittest.TestCase):
  def setUp(self):
    if not os.path.isdir(TMP_DIR):
      os.makedirs(TMP_DIR)

  def tearDown(self):
    if os.path.isdir(TMP_DIR):
      shutil.rmtree(TMP_DIR)

  def test_1_group_1_weight(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float32')
        }]
    ]

    manifest_json = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=4 * 4)
    manifest = json.loads(manifest_json)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')

    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of1'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'float32'
            }]
        }])

    weights_path = os.path.join(TMP_DIR, 'group1-shard1of1')
    weight1 = np.fromfile(weights_path, 'float32')
    np.testing.assert_array_equal(weight1, np.array([1, 2, 3], 'float32'))

  def test_1_group_1_weight_sharded(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float32')
        }]
    ]
    # Shard size is smaller than the size of the array so it gets split between
    # multiple files.
    manifest_json = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=2 * 4)
    manifest = json.loads(manifest_json)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')

    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of2', 'group1-shard2of2'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'float32'
            }]
        }])

    shard_1_path = os.path.join(TMP_DIR, 'group1-shard1of2')
    shard_1 = np.fromfile(shard_1_path, 'float32')
    np.testing.assert_array_equal(shard_1, np.array([1, 2], 'float32'))

    shard_2_path = os.path.join(TMP_DIR, 'group1-shard2of2')
    shard_2 = np.fromfile(shard_2_path, 'float32')
    np.testing.assert_array_equal(shard_2, np.array([3], 'float32'))

  def test_1_group_2_weights_packed(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float32')
        }, {
            'name': 'weight2',
            'data': np.array([4, 5], 'float32')
        }]
    ]

    # Shard size is larger than the sum of the two weights so they get packed.
    manifest_json = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=8 * 4)
    manifest = json.loads(manifest_json)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')
    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of1'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'float32'
            }, {
                'name': 'weight2',
                'shape': [2],
                'dtype': 'float32'
            }]
        }])

    weights_path = os.path.join(TMP_DIR, 'group1-shard1of1')
    weights = np.fromfile(weights_path, 'float32')
    np.testing.assert_array_equal(weights, np.array([1, 2, 3, 4, 5], 'float32'))

  def test_1_group_2_packed_sharded_multi_dtype(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'int32')
        }, {
            'name': 'weight2',
            'data': np.array([4.1, 5.1], 'float32')
        }]
    ]

    # Shard size is smaller than the sum of the weights so they get packed and
    # then sharded. The two buffers will get sharded into 3 files, with shapes
    # [2], [2], and [1]. The second shard is a mixed dtype.
    manifest_json = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=2 * 4)
    manifest = json.loads(manifest_json)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')
    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of3',
                      'group1-shard2of3',
                      'group1-shard3of3'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'int32'
            }, {
                'name': 'weight2',
                'shape': [2],
                'dtype': 'float32'
            }]
        }])

    shard_1_path = os.path.join(TMP_DIR, 'group1-shard1of3')
    shard_1 = np.fromfile(shard_1_path, 'int32')
    np.testing.assert_array_equal(shard_1, np.array([1, 2], 'int32'))

    # Shard 2 has a mixed dtype so we parse the bytes directly.
    shard_2_path = os.path.join(TMP_DIR, 'group1-shard2of3')
    with open(shard_2_path, 'rb') as f:
      shard_2_bytes = f.read()
    shard_2_int = np.frombuffer(shard_2_bytes[:4], 'int32')
    np.testing.assert_array_equal(shard_2_int, np.array([3], 'int32'))
    shard_2_float = np.frombuffer(shard_2_bytes[4:], 'float32')
    np.testing.assert_array_equal(shard_2_float, np.array([4.1], 'float32'))

    shard_3_path = os.path.join(TMP_DIR, 'group1-shard3of3')
    shard_3 = np.fromfile(shard_3_path, 'float32')
    np.testing.assert_array_equal(shard_3, np.array([5.1], 'float32'))

  def test_2_groups_4_weights_sharded_packed(self):
    groups = [
        # Group 1
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float32')
        }, {
            'name': 'weight2',
            'data': np.array([[4, 5], [6, 7]], 'float32')
        }],
        # Group 2
        [{
            'name': 'weight3',
            'data': np.array([1, 2, 3, 4], 'int32')
        }, {
            'name': 'weight4',
            'data': np.array([[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]], 'float32')
        }]
    ]

    manifest_json = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=4 * 4)
    manifest = json.loads(manifest_json)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')
    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of2', 'group1-shard2of2'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'float32'
            }, {
                'name': 'weight2',
                'shape': [2, 2],
                'dtype': 'float32'
            }]
        }, {
            'paths': ['group2-shard1of3',
                      'group2-shard2of3',
                      'group2-shard3of3'],
            'weights': [{
                'name': 'weight3',
                'shape': [4],
                'dtype': 'int32'
            }, {
                'name': 'weight4',
                'shape': [3, 2],
                'dtype': 'float32'
            }]
        }])

    group0_shard_1_path = os.path.join(TMP_DIR, 'group1-shard1of2')
    group0_shard_1 = np.fromfile(group0_shard_1_path, 'float32')
    np.testing.assert_array_equal(
        group0_shard_1, np.array([1, 2, 3, 4], 'float32'))

    group0_shard_2_path = os.path.join(TMP_DIR, 'group1-shard2of2')
    group0_shard_2 = np.fromfile(group0_shard_2_path, 'float32')
    np.testing.assert_array_equal(
        group0_shard_2, np.array([5, 6, 7], 'float32'))

    group1_shard_1_path = os.path.join(TMP_DIR, 'group2-shard1of3')
    group1_shard_1 = np.fromfile(group1_shard_1_path, 'int32')
    np.testing.assert_array_equal(
        group1_shard_1, np.array([1, 2, 3, 4], 'int32'))

    group2_shard_2_path = os.path.join(TMP_DIR, 'group2-shard2of3')
    group2_shard_2 = np.fromfile(group2_shard_2_path, 'float32')
    np.testing.assert_array_equal(
        group2_shard_2, np.array([1.1, 1.2, 1.3, 1.4], 'float32'))

    group2_shard_3_path = os.path.join(TMP_DIR, 'group2-shard3of3')
    group2_shard_3 = np.fromfile(group2_shard_3_path, 'float32')
    np.testing.assert_array_equal(
        group2_shard_3, np.array([1.5, 1.6], 'float32'))

  def test_no_write_manfest(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float32')
        }]
    ]

    manifest_json = write_weights.write_weights(
        groups, TMP_DIR, write_manifest=False)
    manifest = json.loads(manifest_json)

    self.assertFalse(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json exists, but expected not to exist')
    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of1'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'float32'
            }]
        }])

    weights_path = os.path.join(TMP_DIR, 'group1-shard1of1')
    weight1 = np.fromfile(weights_path, 'float32')
    np.testing.assert_array_equal(weight1, np.array([1, 2, 3], 'float32'))

  def test_no_weights_groups_throws(self):
    groups = None
    with self.assertRaises(Exception):
      write_weights.write_weights(groups, TMP_DIR)

  def test_empty_groups_throws(self):
    groups = []
    with self.assertRaises(Exception):
      write_weights.write_weights(groups, TMP_DIR)

  def test_non_grouped_weights_throws(self):
    groups = [{
        'name': 'weight1',
        'data': np.array([1, 2, 3], 'float32')
    }]

    with self.assertRaises(Exception):
      write_weights.write_weights(groups, TMP_DIR)

  def test_bad_weights_entry_throws_no_name(self):
    groups = [
        [{
            'noname': 'weight1',
            'data': np.array([1, 2, 3], 'float32')
        }]
    ]

    with self.assertRaises(Exception):
      write_weights.write_weights(groups, TMP_DIR)

  def test_bad_weights_entry_throws_no_data(self):
    groups = [
        [{
            'name': 'weight1',
            'nodata': np.array([1, 2, 3], 'float32')
        }]
    ]

    with self.assertRaises(Exception):
      write_weights.write_weights(groups, TMP_DIR)

  def test_bad_numpy_array_dtype_throws(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float64')
        }]
    ]

    with self.assertRaises(Exception):
      write_weights.write_weights(groups, TMP_DIR)

  def test_duplicate_weight_name_throws(self):
    groups = [
        [{
            'name': 'duplicate',
            'data': np.array([1, 2, 3], 'float32')
        }], [{
            'name': 'duplicate',
            'data': np.array([4, 5, 6], 'float32')
        }]
    ]

    with self.assertRaises(Exception):
      write_weights.write_weights(groups, TMP_DIR)

  def test_quantize_group(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float32')
        }, {
            'name': 'weight2',
            'data': np.array([4, 5], 'int32')
        }]
    ]

    manifest_json = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=8 * 4, quantization_dtype=np.uint8)
    manifest = json.loads(manifest_json)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')
    q, s, m = zip(
        quantization.quantize_weights(groups[0][0]['data'], np.uint8),
        quantization.quantize_weights(groups[0][1]['data'], np.uint8))
    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of1'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'float32',
                'quantization': {
                    'min': m[0], 'scale': s[0], 'dtype': 'uint8'
                }
            }, {
                'name': 'weight2',
                'shape': [2],
                'dtype': 'int32',
                'quantization': {
                    'min': m[1], 'scale': s[1], 'dtype': 'uint8'
                }
            }]
        }])

    weights_path = os.path.join(TMP_DIR, 'group1-shard1of1')
    weights = np.fromfile(weights_path, 'uint8')
    np.testing.assert_array_equal(weights, np.concatenate([q[0], q[1]]))


if __name__ == '__main__':
  unittest.main()
