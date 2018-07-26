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
from tensorflowjs import quantization

_OUTPUT_DTYPES = [np.float32, np.int32, np.uint8, np.uint16, np.bool]

def write_weights(
    weight_groups, write_dir, shard_size_bytes=1024 * 1024 * 4,
    write_manifest=True, quantization_dtype=None):
  """Writes weights to a binary format on disk for ingestion by JavaScript.

    Weights are organized into groups. When writing to disk, the bytes from all
    weights in each group are concatenated together and then split into shards
    (default is 4MB). This means that large weights (> shard_size) get sharded
    and small weights (< shard_size) will be packed. If the bytes can't be split
    evenly into shards, there will be a leftover shard that is smaller than the
    shard size.

    Weights are optionally quantized to either 8 or 16 bits for compression,
    which is enabled via the `quantization_dtype` argument.

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
      quantization_dtype: An optional numpy dtype to quantize weights to for
        compression. Only np.uint8 and np.uint16 are supported.
    Returns:
      The weights manifest JSON string.

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
          'quantization': {'min': -2.4, 'scale': 0.08, 'dtype': 'uint8'}
        }]
      }]
  """
  _assert_weight_groups_valid(weight_groups)
  _assert_shard_size_bytes_valid(shard_size_bytes)
  _assert_no_duplicate_weight_names(weight_groups)

  manifest = []

  for group_index, group in enumerate(weight_groups):
    if quantization_dtype:
      group = [_quantize_entry(e, quantization_dtype) for e in group]
    group_bytes, total_bytes, _ = _stack_group_bytes(group)

    shard_filenames = _shard_group_bytes_to_disk(
        write_dir, group_index, group_bytes, total_bytes, shard_size_bytes)

    weights_entries = _get_weights_manifest_for_group(group)
    manifest_entry = {
        'paths': shard_filenames,
        'weights': weights_entries
    }
    manifest.append(manifest_entry)

  manifest_json = json.dumps(manifest)

  if write_manifest:
    manifest_path = os.path.join(write_dir, 'weights_manifest.json')
    with open(manifest_path, 'wb') as f:
      f.write(manifest_json.encode())

  return manifest_json

def _quantize_entry(entry, quantization_dtype):
  """Quantizes the weights in the entry, returning a new entry.

  The weights are quantized by linearly re-scaling the values between the
  minimum and maximum value, and representing them with the number of bits
  provided by the `quantization_dtype`.

  In order to guarantee that 0 is perfectly represented by one of the quanzitzed
  values, the range is "nudged" in the same manner as in TF-Lite.

  Args:
    entry: A weight entries to quantize.
    quantization_dtype: An numpy dtype to quantize weights to. Only np.uint8 and
      np.uint16 are supported.

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
                           'original_dtype': 'float32'}
        }
  """
  data = entry['data']
  quantized_data, scale, min_val = quantization.quantize_weights(
      data, quantization_dtype)
  quantized_entry = entry.copy()
  quantized_entry['data'] = quantized_data
  quantized_entry['quantization'] = {
      'min': min_val, 'scale': scale, 'original_dtype': data.dtype.name}
  return quantized_entry

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
    data_bytes = data.tobytes()
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

    filename = ('group' + str(group_index + 1) +
                '-shard' + str(i + 1) + 'of' + str(num_shards))
    filenames.append(filename)
    filepath = os.path.join(write_dir, filename)

    # Write the shard to disk.
    with open(filepath, 'wb') as f:
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
    if is_quantized:
      var_manifest['quantization'] = {
          'min': entry['quantization']['min'],
          'scale': entry['quantization']['scale'],
          'dtype': entry['data'].dtype.name
      }
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


def _assert_valid_weight_entry(entry):
  if not 'name' in entry:
    raise ValueError('Error dumping weight, no name field found.')
  if not 'data' in entry:
    raise ValueError('Error dumping weight, no data field found.')

  name = entry['name']
  data = entry['data']

  if not data.dtype in _OUTPUT_DTYPES:
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
            'weight_groups[' + i + '][' + j + '] has no numpy ' + \
            'array field \'data\'')
      if not isinstance(weights['data'], np.ndarray):
        raise ValueError(
            'weight_groups[' + i + '][' + j + '][\'data\'] is not a numpy ' + \
            'array')


def _assert_shard_size_bytes_valid(shard_size_bytes):
  if shard_size_bytes < 0:
    raise ValueError(
        'shard_size_bytes must be greater than 0, but got ' + shard_size_bytes)
  if not isinstance(shard_size_bytes, int):
    raise ValueError(
        'shard_size_bytes must be an integer, but got ' + shard_size_bytes)
