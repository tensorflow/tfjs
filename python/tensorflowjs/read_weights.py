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
"""Read weights stored in TensorFlow.js-format binary files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os

import numpy as np
from tensorflowjs import quantization

_INPUT_DTYPES = [np.float32, np.int32, np.uint8, np.uint16]


def read_weights(weights_manifest, base_path, flatten=False):
  """Load weight values according to a TensorFlow.js weights manifest.

  Args:
    weights_manifest: A TensorFlow.js-format weights manifest (a JSON array).
    base_path: Base path prefix for the weights files.
    flatten: Whether all the weight groups in the return value are to be
      flattened as a single weights group. Default: `False`.

  Returns:
    If `flatten` is `False`, a `list` of weight groups. Each group is an array
    of weight entries. Each entry is a dict that maps a unique name to a numpy
    array, for example:
        entry = {
          'name': 'weight1',
          'data': np.array([1, 2, 3], 'float32')
        }

        Weights groups would then look like:
        weight_groups = [
          [group_0_entry1, group_0_entry2],
          [group_1_entry1, group_1_entry2],
        ]
    If `flatten` is `True`, returns a single weight group.
  """
  if not isinstance(weights_manifest, list):
    raise ValueError(
        'weights_manifest should be a `list`, but received %s' %
        type(weights_manifest))

  data_buffers = []
  for group in weights_manifest:
    buff = io.BytesIO()
    buff_writer = io.BufferedWriter(buff)
    for path in group['paths']:
      with open(os.path.join(base_path, path), 'rb') as f:
        buff_writer.write(f.read())
    buff_writer.flush()
    buff_writer.seek(0)
    data_buffers.append(buff.read())
  return decode_weights(weights_manifest, data_buffers, flatten=flatten)


def decode_weights(weights_manifest, data_buffers, flatten=False):
  """Load weight values from buffer(s) according to a weights manifest.

  Args:
    weights_manifest: A TensorFlow.js-format weights manifest (a JSON array).
    data_buffers: A buffer or a `list` of buffers containing the weights values
      in binary format, concatenated in the order specified in
      `weights_manifest`. If a `list` of buffers, the length of the `list`
      must match the length of `weights_manifest`. A single buffer is
      interpreted as a `list` of one buffer and is valid only if the length of
      `weights_manifest` is `1`.
    flatten: Whether all the weight groups in the return value are to be
      flattened as a single weight groups. Default: `False`.

  Returns:
    If `flatten` is `False`, a `list` of weight groups. Each group is an array
    of weight entries. Each entry is a dict that maps a unique name to a numpy
    array, for example:
        entry = {
          'name': 'weight1',
          'data': np.array([1, 2, 3], 'float32')
        }

        Weights groups would then look like:
        weight_groups = [
          [group_0_entry1, group_0_entry2],
          [group_1_entry1, group_1_entry2],
        ]
    If `flatten` is `True`, returns a single weight group.

  Raises:
    ValueError: if the lengths of `weights_manifest` and `data_buffers` do not
      match.
  """
  if not isinstance(data_buffers, list):
    data_buffers = [data_buffers]
  if len(weights_manifest) != len(data_buffers):
    raise ValueError(
        'Mismatch in the length of weights_manifest (%d) and the length of '
        'data buffers (%d)' % (len(weights_manifest), len(data_buffers)))

  out = []
  for group, data_buffer in zip(weights_manifest, data_buffers):
    offset = 0
    out_group = []

    for weight in group['weights']:
      quantization_info = weight.get('quantization', None)
      name = weight['name']
      dtype = np.dtype(
          quantization_info['dtype'] if quantization_info else weight['dtype'])
      shape = weight['shape']
      if dtype not in _INPUT_DTYPES:
        raise NotImplementedError('Unsupported data type: %s' % dtype)
      unit_bytes = dtype.itemsize

      weight_numel = 1
      for dim in shape:
        weight_numel *= dim
      weight_bytes = unit_bytes * weight_numel
      value = np.frombuffer(
          data_buffer, dtype=dtype, count=weight_numel,
          offset=offset).reshape(shape)
      if quantization_info:
        value = quantization.dequantize_weights(
            value, quantization_info['scale'], quantization_info['min'],
            np.dtype(weight['dtype']))
      offset += weight_bytes
      out_group.append({'name': name, 'data': value})

    if flatten:
      out += out_group
    else:
      out.append(out_group)

  return out
