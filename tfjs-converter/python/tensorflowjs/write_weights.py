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

import io
import json
import math
import os

import numpy as np
import tensorflow as tf

from tensorflowjs import quantization
from tensorflowjs import read_weights

_OUTPUT_DTYPES = [np.float16, np.float32, np.int32, np.complex64,
                  np.uint8, np.uint16, np.bool, np.object]
_AUTO_DTYPE_CONVERSION = {
    np.dtype(np.float16): np.float32,
    np.dtype(np.float64): np.float32,
    np.dtype(np.int64): np.int32,
    np.dtype(np.complex128): np.complex64}

def write_weights(
    weight_groups, write_dir, shard_size_bytes=1024 * 1024 * 4,
    write_manifest=True, quantization_dtype_map=None):
  """Writes weights to a binary format on disk for ingestion by JavaScript.

    Weights are organized into groups. When writing to disk, the bytes from all
    weights in each group are concatenated together and then split into shards
    (default is 4MB). This means that large weights (> shard_size) get sharded
    and small weights (< shard_size) will be packed. If the bytes can't be split
    evenly into shards, there will be a leftover shard that is smaller than the
    shard size.

    Weights are optionally quantized to either 8 or 16 bits for compression,
    which is enabled via the `quantization_dtype_map`.

    Args:
      weight_groups: An list of groups. Each group is an array of weight
        entries. Each entry is a dict that maps a unique name to a numpy array,
        for example:
        entry = {
          'name': 'weight1',
          'data': np.array([1, 2, 3], 'float32')
        }

        Weights groups would then look like:
        weight_groups = [
          [group_0_entry1, group_0_entry2],
          [group_1_entry1, group_1_entry2],
        ]

        The 'name' must be unique across all groups and all entries. The 'data'
        field must be a numpy ndarray.
      write_dir: A directory to write the files to.
      shard_size_bytes: The size of shards in bytes. Defaults to 4MB, which is
        the max file size for caching for all major browsers.
      write_manifest: Whether to write the manifest JSON to disk. Defaults to
        True.
      quantization_dtype_map: (Optional) A mapping from dtype
        (`uint8`, `uint16`, `float16`) to weights names. The weight mapping
        supports wildcard substitution.
    Returns:
      The weights manifest JSON dict.

      An example manifest with 2 groups, 2 weights, and each weight sharded
      into 2:

      The manifest JSON looks like the following:
      [{
        'paths': ['group1-shard1of2', 'group1-shard2of2'],
        'weights': [{
          'name': 'weight1',
          'shape': [1000, 1000],
          'dtype': 'float32'
        }]
      }, {
        'paths': ['group2-shard1of2', 'group2-shard2of2'],
        'weights': [{
          'name': 'weight2',
          'shape': [2000, 2000],
          'dtype': 'float32'
        }]
      }]
      or, if quantization is used:
      [{
        'paths': ['group1-shard1of2', 'group1-shard2of2'],
        'weights': [{
          'name': 'weight1',
          'shape': [1000, 1000],
          'dtype': 'float32'
          'quantization': {'min': -0.1, 'scale': 0.01, 'dtype': 'uint8'}
        }]
      }, {
        'paths': ['group2-shard1of2', 'group2-shard2of2'],
        'weights': [{
          'name': 'weight2',
          'shape': [2000, 2000],
          'dtype': 'float32',
          'quantization': {'dtype': 'float16'}
        }]
      }]
  """
  _assert_weight_groups_valid(weight_groups)
  _assert_shard_size_bytes_valid(shard_size_bytes)
  _assert_no_duplicate_weight_names(weight_groups)

  manifest = []

  for group_index, group in enumerate(weight_groups):
    for e in group:
      _auto_convert_weight_entry(e)
    names = [entry['name'] for entry in group]
    quantization_dtype = quantization.map_layers_to_quantization_dtype(
        names, quantization_dtype_map)

    group = [
        _quantize_entry(e, quantization_dtype[e['name']])
        if e['name'] in quantization_dtype else e for e in group
    ]
    group_bytes, total_bytes, _ = _stack_group_bytes(group)

    shard_filenames = _shard_group_bytes_to_disk(
        write_dir, group_index, group_bytes, total_bytes, shard_size_bytes)

    weights_entries = _get_weights_manifest_for_group(group)
    manifest_entry = {
        'paths': shard_filenames,
        'weights': weights_entries
    }
    manifest.append(manifest_entry)

  if write_manifest:
    manifest_path = os.path.join(write_dir, 'weights_manifest.json')
    with tf.io.gfile.GFile(manifest_path, 'wb') as f:
      f.write(json.dumps(manifest).encode())

  return manifest


def _quantize_entry(entry, quantization_dtype):
  """Quantizes the weights in the entry, returning a new entry.

  The weights are quantized by linearly re-scaling the values between the
  minimum and maximum value, and representing them with the number of bits
  provided by the `quantization_dtype`.

  In order to guarantee that 0 is perfectly represented by one of the quanzitzed
  values, the range is "nudged" in the same manner as in TF-Lite.

  Args:
    entry: A weight entries to quantize.
    quantization_dtype: An numpy dtype to quantize weights to.
        Only np.uint8, np.uint16, and np.float16 are supported.

  Returns:
    A new entry containing the quantized data and additional quantization info,
    for example:
        original_entry = {
          'name': 'weight1',
          'data': np.array([0, -0.1, 1.2], 'float32')
        }
        quantized_entry = {
          'name': 'weight1',
          'data': np.array([20, 0, 255], 'uint8')
          'quantization': {'min': -0.10196078817, 'scale': 0.00509803940852,
                           'dtype': 'uint8', 'original_dtype': 'float32'}
        }
  """
  data = entry['data']
  # Only float32 tensors are quantized.
  if data.dtype != 'float32':
    return entry
  quantized_data, metadata = quantization.quantize_weights(
      data, quantization_dtype)
  metadata.update({'original_dtype': data.dtype.name})
  quantized_entry = entry.copy()
  quantized_entry['data'] = quantized_data
  quantized_entry['quantization'] = metadata
  return quantized_entry


def _serialize_string_array(data):
  """Serializes a numpy array of dtype `string` into bytes.

  Each string value is preceded by 4 bytes which denote a 32-bit unsigned
  integer in little endian that specifies the byte length of the following
  string. This is followed by the actual string bytes. If the tensor has no
  strings there will be no bytes reserved. Empty strings will still take 4 bytes
  for the length.

  For example, a tensor that has 2 strings will be encoded as
  [byte length of s1][bytes of s1...][byte length of s2][bytes of s2...]

  where byte length always takes 4 bytes.

  Args:
    data: A numpy array of dtype `string`.

  Returns:
    bytes of the entire string tensor to be serialized on disk.
  """
  strings = data.flatten().tolist()

  string_bytes = io.BytesIO()
  bytes_writer = io.BufferedWriter(string_bytes)

  for x in strings:
    encoded = x if isinstance(x, bytes) else x.encode('utf-8')
    length_as_bytes = np.array(len(encoded),
                               read_weights.STRING_LENGTH_DTYPE).tobytes()
    bytes_writer.write(length_as_bytes)
    bytes_writer.write(encoded)
  bytes_writer.flush()
  string_bytes.seek(0)
  return string_bytes.read()

def _serialize_numeric_array(data):
  """Serializes a numeric numpy array into bytes.

  Args:
    data: A numeric numpy array.

  Returns:
    bytes of the array to be serialized on disk.
  """
  return data.tobytes()

def _stack_group_bytes(group):
  """Stacks the bytes for a weight group into a flat byte array.

  Args:
    group: A list of weight entries.
  Returns:
    A type: (group_bytes, total_bytes, weights_entries, group_bytes_writer)
    group_bytes: The stacked bytes for the group, as a BytesIO() stream.
    total_bytes: A number representing the total size of the byte buffer.
    groups_bytes_writer: The io.BufferedWriter object. Returned so that
      group_bytes does not get garbage collected and closed.

  """
  group_bytes = io.BytesIO()
  group_bytes_writer = io.BufferedWriter(group_bytes)
  total_bytes = 0

  for entry in group:
    _assert_valid_weight_entry(entry)
    data = entry['data']

    if data.dtype == np.object:
      data_bytes = _serialize_string_array(data)
    else:
      data_bytes = _serialize_numeric_array(data)
    group_bytes_writer.write(data_bytes)
    total_bytes += len(data_bytes)

  group_bytes_writer.flush()
  group_bytes.seek(0)

  # NOTE: We must return the bytes writer here, otherwise it goes out of scope
  # and python closes the IO operation.
  return (group_bytes, total_bytes, group_bytes_writer)


def _shard_group_bytes_to_disk(
    write_dir, group_index, group_bytes, total_bytes, shard_size_bytes):
  """Shards the concatenated bytes for a group to disk.

  Args:
    write_dir: The directory to write the files to.
    group_index: The index for the group.
    group_bytes: An io.BytesIO() object representing the byte array.
    total_bytes: The total number of bytes of the stream.
    shard_size_bytes: The size of shards in bytes. If None, the whole byte
        array will be written as one shard.
  Returns:
    A list of filenames that were written to disk.
  """
  if shard_size_bytes is None:
    shard_size_bytes = total_bytes

  num_shards = int(math.ceil(float(total_bytes) / shard_size_bytes))

  filenames = []
  for i in range(num_shards):
    shard = group_bytes.read(shard_size_bytes)

    filename = 'group%d-shard%dof%d.bin' % (group_index + 1, i + 1, num_shards)
    filenames.append(filename)
    filepath = os.path.join(write_dir, filename)

    # Write the shard to disk.
    with tf.io.gfile.GFile(filepath, 'wb') as f:
      f.write(shard)

  return filenames


def _get_weights_manifest_for_group(group):
  """Gets the weights entries manifest JSON for a group.

  Args:
    group: A list of weight entries.
  Returns:
    An list of manifest entries (dicts) to be written in the weights manifest.
  """
  weights_entries = []
  for entry in group:
    is_quantized = 'quantization' in entry
    dtype = (entry['quantization']['original_dtype']
             if is_quantized else entry['data'].dtype.name)
    var_manifest = {
        'name': entry['name'],
        'shape': list(entry['data'].shape),
        'dtype': dtype
    }
    # String arrays have dtype 'object' and need extra metadata to parse.
    if dtype == 'object':
      var_manifest['dtype'] = 'string'
    if is_quantized:
      manifest = {'dtype': entry['data'].dtype.name}
      manifest.update(entry['quantization'])
      var_manifest['quantization'] = manifest
    weights_entries.append(var_manifest)
  return weights_entries


def _assert_no_duplicate_weight_names(weight_groups):
  weight_names = set()
  for group in weight_groups:
    for entry in group:
      name = entry['name']
      if name in weight_names:
        raise Exception(
            'Error dumping weights, duplicate weight name ' + name)
      weight_names.add(name)


def _auto_convert_weight_entry(entry):
  data = entry['data']
  if data.dtype in _AUTO_DTYPE_CONVERSION:
    entry['data'] = data.astype(_AUTO_DTYPE_CONVERSION[data.dtype])
    print('weight ' + entry['name'] + ' with shape ' + str(data.shape) +
          ' and dtype ' + data.dtype.name + ' was auto converted to the type ' +
          np.dtype(_AUTO_DTYPE_CONVERSION[data.dtype]).name)


def _assert_valid_weight_entry(entry):
  if 'name' not in entry:
    raise ValueError('Error dumping weight, no name field found.')
  if 'data' not in entry:
    raise ValueError('Error dumping weight, no data field found.')

  name = entry['name']
  data = entry['data']

  # String tensors can be backed by different numpy dtypes, thus we consolidate
  # to a single 'np.object' dtype.
  if data.dtype.name.startswith('str') or data.dtype.name.startswith('bytes'):
    data = data.astype(np.object)
    entry['data'] = data


  if not (data.dtype in _OUTPUT_DTYPES or data.dtype in _AUTO_DTYPE_CONVERSION):
    raise ValueError('Error dumping weight ' + name + ', dtype ' +
                     data.dtype.name + ' not supported.')

  if not isinstance(data, np.ndarray):
    raise ValueError('Error dumping weight ' + name + ', data ' +
                     'must be a numpy ndarray.')


def _assert_weight_groups_valid(weight_groups):
  if not isinstance(weight_groups, list):
    raise Exception('weight_groups must be a list of groups')
  if not weight_groups:
    raise ValueError('weight_groups must have more than one list element')
  for i, weight_group in enumerate(weight_groups):
    if not isinstance(weight_group, list):
      raise ValueError(
          'weight_groups[' + i + '] must be a list of weight entries')
    for j, weights in enumerate(weight_group):
      if 'name' not in weights:
        raise ValueError(
            'weight_groups[' + i + '][' + j + '] has no string field \'name\'')
      if 'data' not in weights:
        raise ValueError(
            'weight_groups[' + i + '][' + j + '] has no numpy ' +
            'array field \'data\'')
      if not isinstance(weights['data'], np.ndarray):
        raise ValueError(
            'weight_groups[' + i + '][' + j + '][\'data\'] is not a numpy ' +
            'array')


def _assert_shard_size_bytes_valid(shard_size_bytes):
  if shard_size_bytes <= 0:
    raise ValueError(
        'shard_size_bytes must be greater than 0, but got %s' %
        shard_size_bytes)
  if not isinstance(shard_size_bytes, int):
    raise ValueError(
        'shard_size_bytes must be an integer, but got %s' %
        shard_size_bytes)
