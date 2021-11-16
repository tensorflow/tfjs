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

import os
import shutil

import numpy as np
import tensorflow as tf

from tensorflowjs import write_weights

TMP_DIR = '/tmp/write_weights_test/'


class TestWriteWeights(tf.test.TestCase):
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

    manifest = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=4 * 4)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')

    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of1.bin'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'float32'
            }]
        }])

    weights_path = os.path.join(TMP_DIR, 'group1-shard1of1.bin')
    weight1 = np.fromfile(weights_path, 'float32')
    np.testing.assert_array_equal(weight1, np.array([1, 2, 3], 'float32'))

  def test_1_group_1_weight_bool(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([True, False, True], 'bool')
        }]
    ]

    manifest = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=4 * 4)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')

    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of1.bin'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'bool'
            }]
        }])

    weights_path = os.path.join(TMP_DIR, 'group1-shard1of1.bin')
    weight1 = np.fromfile(weights_path, 'bool')
    np.testing.assert_array_equal(
        weight1, np.array([True, False, True], 'bool'))

  def test_1_group_1_weight_string(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([['здраво', 'end'], ['test', 'a']], 'object')
        }]
    ]

    manifest = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=4 * 1024 * 1024)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')

    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of1.bin'],
            'weights': [{
                'name': 'weight1',
                'shape': [2, 2],
                'dtype': 'string'
            }]
        }])

    weights_path = os.path.join(TMP_DIR, 'group1-shard1of1.bin')
    with open(weights_path, 'rb') as f:
      weight_bytes = f.read()

      self.assertEqual(len(weight_bytes), 36)
      # 'здраво'
      size = np.frombuffer(weight_bytes[:4], 'uint32')[0]
      self.assertEqual(size, 12)  # 6 cyrillic chars (2 bytes each).
      string = weight_bytes[4:16].decode('utf-8')
      self.assertEqual(string, u'здраво')
      # 'end'
      size = np.frombuffer(weight_bytes[16:20], 'uint32')[0]
      self.assertEqual(size, 3)  # 3 ascii chars.
      string = weight_bytes[20:23].decode('utf-8')
      self.assertEqual(string, u'end')
      # 'test'
      size = np.frombuffer(weight_bytes[23:27], 'uint32')[0]
      self.assertEqual(size, 4)  # 4 ascii chars.
      string = weight_bytes[27:31].decode('utf-8')
      self.assertEqual(string, u'test')
      # 'a'
      size = np.frombuffer(weight_bytes[31:35], 'uint32')[0]
      self.assertEqual(size, 1)  # 4 ascii chars.
      string = weight_bytes[35:36].decode('utf-8')
      self.assertEqual(string, u'a')


  def test_1_group_1_weight_string_empty(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([''], 'object')
        }]
    ]

    manifest = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=4 * 1024 * 1024)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')

    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of1.bin'],
            'weights': [{
                'name': 'weight1',
                'shape': [1],
                'dtype': 'string'
            }]
        }])

    weights_path = os.path.join(TMP_DIR, 'group1-shard1of1.bin')
    with open(weights_path, 'rb') as f:
      weight_bytes = f.read()
      self.assertEqual(len(weight_bytes), 4)
      size = np.frombuffer(weight_bytes[:4], 'uint32')[0]
      self.assertEqual(size, 0)  # Empty string.

  def test_1_group_1_weight_string_unicode(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([[u'здраво', u'end'], [u'test', u'a']], 'object')
        }]
    ]

    manifest = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=4 * 1024 * 1024)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')

    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of1.bin'],
            'weights': [{
                'name': 'weight1',
                'shape': [2, 2],
                'dtype': 'string'
            }]
        }])

    weights_path = os.path.join(TMP_DIR, 'group1-shard1of1.bin')
    with open(weights_path, 'rb') as f:
      weight_bytes = f.read()

      self.assertEqual(len(weight_bytes), 36)
      # 'здраво'
      size = np.frombuffer(weight_bytes[:4], 'uint32')[0]
      self.assertEqual(size, 12)  # 6 cyrillic chars (2 bytes each).
      string = weight_bytes[4:16].decode('utf-8')
      self.assertEqual(string, u'здраво')
      # 'end'
      size = np.frombuffer(weight_bytes[16:20], 'uint32')[0]
      self.assertEqual(size, 3)  # 3 ascii chars.
      string = weight_bytes[20:23].decode('utf-8')
      self.assertEqual(string, u'end')
      # 'test'
      size = np.frombuffer(weight_bytes[23:27], 'uint32')[0]
      self.assertEqual(size, 4)  # 4 ascii chars.
      string = weight_bytes[27:31].decode('utf-8')
      self.assertEqual(string, u'test')
      # 'a'
      size = np.frombuffer(weight_bytes[31:35], 'uint32')[0]
      self.assertEqual(size, 1)  # 4 ascii chars.
      string = weight_bytes[35:36].decode('utf-8')
      self.assertEqual(string, u'a')

  def test_1_group_1_weight_string_sharded(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array(['helloworld'], 'object')
        }]
    ]

    # The array takes up 14 bytes across 3 shards when shard size is 5 bytes.
    manifest = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=5)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')

    self.assertEqual(
        manifest,
        [{
            'paths': [
                'group1-shard1of3.bin',
                'group1-shard2of3.bin',
                'group1-shard3of3.bin'
            ],
            'weights': [{
                'name': 'weight1',
                'shape': [1],
                'dtype': 'string'
            }]
        }])

    weight_bytes = bytes()
    with open(os.path.join(TMP_DIR, 'group1-shard1of3.bin'), 'rb') as f:
      weight_bytes += f.read()
    with open(os.path.join(TMP_DIR, 'group1-shard2of3.bin'), 'rb') as f:
      weight_bytes += f.read()
    with open(os.path.join(TMP_DIR, 'group1-shard3of3.bin'), 'rb') as f:
      weight_bytes += f.read()

    self.assertEqual(len(weight_bytes), 14)
    size = np.frombuffer(weight_bytes[:4], 'uint32')[0]
    self.assertEqual(size, 10)  # 10 ascii chars.
    string = weight_bytes[4:14].decode('utf-8')
    self.assertEqual(string, u'helloworld')

  def test_1_group_1_weight_complex(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1 + 1j, 2 + 2j, 3 + 3j], 'complex')
        }]
    ]

    manifest = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=6 * 4)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')

    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of1.bin'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'complex64'
            }]
        }])

    weights_path = os.path.join(TMP_DIR, 'group1-shard1of1.bin')
    weight1 = np.fromfile(weights_path, 'complex64')
    np.testing.assert_array_equal(
        weight1, np.array([1 + 1j, 2 + 2j, 3 + 3j], 'complex64'))

  def test_1_group_3_weights_packed_multi_dtype(self):
    # Each string tensor uses different encoding.
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float32')
        }, {
            'name': 'weight2',
            'data': np.array([
                u'hello'.encode('utf-16'), u'end'.encode('utf-16')], 'object')
        }, {
            'name': 'weight3',
            'data': np.array([u'здраво'.encode('windows-1251')], 'object')
        }, {
            'name': 'weight4',
            'data': np.array([u'语言处理'.encode('utf-8')], 'object')
        }, {
            'name': 'weight5',
            'data': np.array([4, 5, 6], 'float32')
        }]
    ]

    manifest = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=4 * 1024 * 1024)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')

    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of1.bin'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'float32'
            }, {
                'name': 'weight2',
                'shape': [2],
                'dtype': 'string'
            }, {
                'name': 'weight3',
                'shape': [1],
                'dtype': 'string'
            }, {
                'name': 'weight4',
                'shape': [1],
                'dtype': 'string'
            }, {
                'name': 'weight5',
                'shape': [3],
                'dtype': 'float32'
            }]
        }])

    weights_path = os.path.join(TMP_DIR, 'group1-shard1of1.bin')
    with open(weights_path, 'rb') as f:
      weight_bytes = f.read()
      self.assertEqual(len(weight_bytes), 78)

      # [1, 2, 3]
      weight1 = np.frombuffer(weight_bytes[:12], 'float32')
      np.testing.assert_array_equal(weight1, np.array([1, 2, 3], 'float32'))

      # 'hello'
      size = np.frombuffer(weight_bytes[12:16], 'uint32')[0]
      self.assertEqual(size, 12)  # 5 ascii chars in utf-16.
      string = weight_bytes[16:28].decode('utf-16')
      self.assertEqual(string, u'hello')

      # 'end'
      size = np.frombuffer(weight_bytes[28:32], 'uint32')[0]
      self.assertEqual(size, 8)  # 3 ascii chars in utf-16.
      string = weight_bytes[32:40].decode('utf-16')
      self.assertEqual(string, u'end')

      # 'здраво'
      size = np.frombuffer(weight_bytes[40:44], 'uint32')[0]
      self.assertEqual(size, 6)  # 6 cyrillic chars in windows-1251.
      string = weight_bytes[44:50].decode('windows-1251')
      self.assertEqual(string, u'здраво')

      # '语言处理'
      size = np.frombuffer(weight_bytes[50:54], 'uint32')[0]
      self.assertEqual(size, 12)  # 4 east asian chars in utf-8.
      string = weight_bytes[54:66].decode('utf-8')
      self.assertEqual(string, u'语言处理')

      weight5 = np.frombuffer(weight_bytes[66:], 'float32')
      np.testing.assert_array_equal(weight5, np.array([4, 5, 6], 'float32'))

  def test_1_group_1_weight_sharded(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float32')
        }]
    ]
    # Shard size is smaller than the size of the array so it gets split between
    # multiple files.
    manifest = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=2 * 4)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')

    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of2.bin', 'group1-shard2of2.bin'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'float32'
            }]
        }])

    shard_1_path = os.path.join(TMP_DIR, 'group1-shard1of2.bin')
    shard_1 = np.fromfile(shard_1_path, 'float32')
    np.testing.assert_array_equal(shard_1, np.array([1, 2], 'float32'))

    shard_2_path = os.path.join(TMP_DIR, 'group1-shard2of2.bin')
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
    manifest = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=8 * 4)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')
    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of1.bin'],
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

    weights_path = os.path.join(TMP_DIR, 'group1-shard1of1.bin')
    weights = np.fromfile(weights_path, 'float32')
    np.testing.assert_array_equal(weights, np.array([1, 2, 3, 4, 5], 'float32'))

  def test_1_group_2_packed_sharded_multi_dtype(self):
    groups = [
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'int32')
        }, {
            'name': 'weight2',
            'data': np.array([True, False, False, True], 'bool')
        }, {
            'name': 'weight3',
            'data': np.array([4.1, 5.1], 'float32')
        }]
    ]

    # Shard size is smaller than the sum of the weights so they get packed and
    # then sharded. The two buffers will get sharded into 3 files, with shapes
    # [2], [2], and [1]. The second shard is a mixed dtype.
    manifest = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=2 * 4)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')
    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of3.bin',
                      'group1-shard2of3.bin',
                      'group1-shard3of3.bin'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'int32'
            }, {
                'name': 'weight2',
                'shape': [4],
                'dtype': 'bool'
            }, {
                'name': 'weight3',
                'shape': [2],
                'dtype': 'float32'
            }]
        }])

    shard_1_path = os.path.join(TMP_DIR, 'group1-shard1of3.bin')
    shard_1 = np.fromfile(shard_1_path, 'int32')
    np.testing.assert_array_equal(shard_1, np.array([1, 2], 'int32'))

    # Shard 2 has a mixed dtype so we parse the bytes directly.
    shard_2_path = os.path.join(TMP_DIR, 'group1-shard2of3.bin')
    with open(shard_2_path, 'rb') as f:
      shard_2_bytes = f.read()
    self.assertEqual(len(shard_2_bytes), 8)
    shard_2_int = np.frombuffer(shard_2_bytes[:4], 'int32')
    np.testing.assert_array_equal(shard_2_int, np.array([3], 'int32'))
    shard_2_bool = np.frombuffer(shard_2_bytes[4:], 'bool')
    np.testing.assert_array_equal(
        shard_2_bool, np.array([True, False, False, True], 'bool'))

    shard_3_path = os.path.join(TMP_DIR, 'group1-shard3of3.bin')
    shard_3 = np.fromfile(shard_3_path, 'float32')
    np.testing.assert_array_equal(shard_3, np.array([4.1, 5.1], 'float32'))

  def test_2_groups_4_weights_sharded_packed(self):
    groups = [
        # Group 1
        [{
            'name': 'weight1',
            'data': np.array([1, 2, 3], 'float64')
        }, {
            'name': 'weight2',
            'data': np.array([[4, 5], [6, 7]], 'float32')
        }, {
            'name': 'weight5',
            'data': np.array([1], 'float16')
        }],
        # Group 2
        [{
            'name': 'weight3',
            'data': np.array([1, 2, 3, 4], 'int64')
        }, {
            'name': 'weight4',
            'data': np.array([[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]], 'float32')
        }]
    ]

    manifest = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=4 * 4)

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')
    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of2.bin', 'group1-shard2of2.bin'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'float32'
            }, {
                'name': 'weight2',
                'shape': [2, 2],
                'dtype': 'float32'
            }, {
                'name': 'weight5',
                'shape': [1],
                'dtype': 'float32'
            }]
        }, {
            'paths': ['group2-shard1of3.bin',
                      'group2-shard2of3.bin',
                      'group2-shard3of3.bin'],
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

    group0_shard_1_path = os.path.join(TMP_DIR, 'group1-shard1of2.bin')
    group0_shard_1 = np.fromfile(group0_shard_1_path, 'float32')
    np.testing.assert_array_equal(
        group0_shard_1, np.array([1, 2, 3, 4], 'float32'))

    group0_shard_2_path = os.path.join(TMP_DIR, 'group1-shard2of2.bin')
    group0_shard_2 = np.fromfile(group0_shard_2_path, 'float32')
    np.testing.assert_array_equal(
        group0_shard_2, np.array([5, 6, 7, 1], 'float32'))

    group1_shard_1_path = os.path.join(TMP_DIR, 'group2-shard1of3.bin')
    group1_shard_1 = np.fromfile(group1_shard_1_path, 'int32')
    np.testing.assert_array_equal(
        group1_shard_1, np.array([1, 2, 3, 4], 'int32'))

    group2_shard_2_path = os.path.join(TMP_DIR, 'group2-shard2of3.bin')
    group2_shard_2 = np.fromfile(group2_shard_2_path, 'float32')
    np.testing.assert_array_equal(
        group2_shard_2, np.array([1.1, 1.2, 1.3, 1.4], 'float32'))

    group2_shard_3_path = os.path.join(TMP_DIR, 'group2-shard3of3.bin')
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

    manifest = write_weights.write_weights(
        groups, TMP_DIR, write_manifest=False)

    self.assertFalse(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json exists, but expected not to exist')
    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of1.bin'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'float32'
            }]
        }])

    weights_path = os.path.join(TMP_DIR, 'group1-shard1of1.bin')
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
        }, {
            'name': 'weight3',
            'data': np.array([6, 7], 'float64')
        }, {
            'name': 'weight4',
            'data': np.array(['hello'], np.object)
        }]
    ]

    manifest = write_weights.write_weights(
        groups, TMP_DIR, shard_size_bytes=1024,
        quantization_dtype_map={
            'float16': 'weight1',
            'uint8': 'weight3'
        })

    self.assertTrue(
        os.path.isfile(os.path.join(TMP_DIR, 'weights_manifest.json')),
        'weights_manifest.json does not exist')
    self.assertEqual(
        manifest,
        [{
            'paths': ['group1-shard1of1.bin'],
            'weights': [{
                'name': 'weight1',
                'shape': [3],
                'dtype': 'float32',
                'quantization': {
                    'original_dtype': 'float32',
                    'dtype': 'float16'
                }
            }, {
                'name': 'weight2',
                'shape': [2],
                'dtype': 'int32'
            }, {
                'name': 'weight3',
                'shape': [2],
                'dtype': 'float32',
                'quantization': {
                    'min': 6.0,
                    'scale': 1/255.0,
                    'original_dtype': 'float32',
                    'dtype': 'uint8'
                }
            }, {
                'name': 'weight4',
                'shape': [1],
                'dtype': 'string'
            }]
        }])

    weights_path = os.path.join(TMP_DIR, 'group1-shard1of1.bin')
    with open(weights_path, 'rb') as f:
      weight_bytes = f.read()
      self.assertEqual(len(weight_bytes), 25)
      w1 = np.frombuffer(weight_bytes[:6], 'float16')
      np.testing.assert_array_equal(w1, np.array([1, 2, 3], 'float16'))

      w2 = np.frombuffer(weight_bytes[6:14], 'int32')
      np.testing.assert_array_equal(w2, np.array([4, 5], 'int32'))

      w3 = np.frombuffer(weight_bytes[14:16], 'uint8')
      np.testing.assert_array_equal(w3, np.array([0, 255], 'uint8'))

      size = np.frombuffer(weight_bytes[16:20], 'uint32')[0]
      self.assertEqual(size, 5)  # 5 ascii letters.
      w4 = weight_bytes[20:].decode('utf-8')
      self.assertEqual(w4, u'hello')


if __name__ == '__main__':
  tf.test.main()
