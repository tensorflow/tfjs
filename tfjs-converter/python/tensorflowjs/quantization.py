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

import fnmatch
import numpy as np

QUANTIZATION_DTYPE_FLOAT16 = 'float16'
QUANTIZATION_DTYPE_UINT8 = 'uint8'
QUANTIZATION_DTYPE_UINT16 = 'uint16'

QUANTIZATION_BYTES_TO_DTYPES = {1: QUANTIZATION_DTYPE_UINT8,
                                2: QUANTIZATION_DTYPE_UINT16}
QUANTIZATION_OPTION_TO_DTYPES = {QUANTIZATION_DTYPE_UINT8: np.uint8,
                                 QUANTIZATION_DTYPE_UINT16: np.uint16,
                                 QUANTIZATION_DTYPE_FLOAT16: np.float16}


def map_layers_to_quantization_dtype(names, quantization_dtype_map):
  """Maps node names to their quantization dtypes.

  Given a quantization_dtype_map which maps dtypes `uint8`, `uint16`, `float16`
  to node patterns, e.g., conv/*/weights we construct a new mapping for each
  individual node name to its dtype, e.g., conv/1/weight -> `uint8`.
  A dtype in the map can also be a boolean, signaling a fallthrough dtype.
  There can only be one fallthrough dtype in the map. A fallthrough dtype
  will convert all weights that don't match any pattern to the provided dtype.

  Args:
    names: Array of node names.
    quantization_dtype_map: A mapping from dtype (`uint8`, `uint16`, `float16`)
      to weights. The weight mapping supports wildcard substitution.

  Returns:
    quantization_dtype: A mapping from each node name which matches
    an entry in quantization_dtype_map to its corresponding dtype.

  Raises:
    ValueError: - If multiple dtypes match the same node name
                - If more than one fallthrough is provided
  """
  if quantization_dtype_map is None:
    return {}

  fallthrough = None
  quantization_dtype = {}
  for dtype_name, patterns in quantization_dtype_map.items():
    # Record fallthrough if there is one
    if isinstance(patterns, bool) and patterns:
      # Only one fallthrough is supported
      if fallthrough is not None:
        raise ValueError(
            'More than one quantization fallthrough provided, '
            'exactly one is supported')
      fallthrough = dtype_name
      continue
    if isinstance(patterns, str):
      patterns = list([patterns])

    # Record matched weights for dtype
    for pattern in patterns:
      for match in fnmatch.filter(names, pattern):
        dtype = QUANTIZATION_OPTION_TO_DTYPES[dtype_name]
        if match in quantization_dtype and quantization_dtype[match] != dtype:
          raise ValueError(
              'Two quantization values %s, %s match the same node %s' %
              (dtype, quantization_dtype[match], match))
        quantization_dtype[match] = dtype

  # Catch all remaining names with fallthrough
  if fallthrough is not None:
    nameset = set(names)
    fallthrough_names = nameset - set(quantization_dtype.keys())
    for name in fallthrough_names:
      quantization_dtype[name] = QUANTIZATION_OPTION_TO_DTYPES[fallthrough]

  return quantization_dtype

def quantize_weights(data, quantization_dtype):
  """Quantizes the weights by linearly re-scaling across available bits.

  The weights are quantized by linearly re-scaling the values between the
  minimum and maximum value, and representing them with the number of bits
  provided by the `quantization_dtype`.

  In order to guarantee that 0 is perfectly represented by one of the quantized
  values, the range is "nudged" in the same manner as in TF-Lite.

  Weights can be de-quantized by multiplying by the returned `scale` and adding
  `min`.

  Args:
    data: A numpy array of dtype 'float32' or 'int32'.
    quantization_dtype: A numpy dtype to quantize weights to. Only np.float16,
      np.uint8, and np.uint16 are supported.

  Returns:
    quantized_data: The quantized weights as a numpy array with dtype
      `quantization_dtype`.
    metadata: A dictionary with the corresponding metadata for the quantization
      type. There is no metadata associated with float16.
      For affine quantization there are two associated metadata values:
        scale: The linearly scaling constant used for quantization.
        min_val: The minimum value of the linear range.
  Raises:
    ValueError: if `quantization_dtype` is not a valid type.
  """
  if quantization_dtype in [np.uint8, np.uint16]:
    # Compute the min and max for the group.
    min_val = data.min().astype(np.float64)
    max_val = data.max().astype(np.float64)
    if min_val == max_val:
      # If there is only a single value, we can represent everything as zeros.
      quantized_data = np.zeros_like(data, dtype=quantization_dtype)
      scale = 1.0
    else:
      # Quantize data.
      scale, min_val, max_val = _get_affine_quantization_range(
          min_val, max_val, quantization_dtype)
      quantized_data = np.round(
          (data.clip(min_val, max_val) - min_val) / scale).astype(
              quantization_dtype)

    return quantized_data, {'min': min_val, 'scale': scale}
  elif quantization_dtype == np.float16:
    if data.dtype != np.float32:
      raise ValueError(
          'Invalid data dtype %r\n'
          'float16 quantization only supports float32 dtype' % data.dtype)
    quantized_data = data.astype(np.float16)
    return quantized_data, {}
  else:
    raise ValueError('Invalid `quantization_dtype`: %r' % quantization_dtype)



def dequantize_weights(data, metadata, original_dtype=np.float32):
  dtype = data.dtype

  if dtype in [np.uint8, np.uint16]:
    if not ('scale' in metadata and 'min' in metadata):
      raise ValueError(
          'Missing metadata min or scale for dtype %s' % dtype.name)
    scale = metadata['scale']
    min_val = metadata['min']
    return np.round(data * scale + min_val).astype(original_dtype)
  elif dtype == np.float16:
    if original_dtype != np.float32:
      raise ValueError(
          'Invalid data dtype %r\n'
          'float16 quantization only supports float32 dtype' % data.dtype)
    return data.astype(original_dtype)
  else:
    raise ValueError(
        'Invalid dtype %s for dequantization\n'
        'Supported dtypes are uint8, uint16, float16' % dtype.name)

def _get_affine_quantization_range(min_val, max_val, quantization_dtype):
  """Computes quantization range to ensure that zero is represented if covered.

  Gymnastics with nudged zero point is to ensure that real zero maps to an
  integer, which is required for e.g. zero-padding in convolutional layers.

  Based on `NudgeQuantizationRange` in
  tensorflow/contrib/lite/kernels/internal/quantization_util.h, except we do not
  nudge if 0 is not in the range.

  Args:
    min_val: The actual minimum value of the data.
    max_val: The actual maximum value of the data.
    quantization_dtype: A numpy dtype to quantize weights to. Only np.uint8 and
      np.uint16 are supported.

  Returns:
    scale: The linear scaling constant used for quantization.
    nudged_min: The adjusted minimum value to ensure zero is represented, if
      covered.
    nudged_max: The adjusted maximum value to ensure zero is represented, if
      covered.
  Raises:
    ValueError: if `quantization_dtype` is not a valid type.
  """
  if quantization_dtype not in [np.uint8, np.uint16]:
    raise ValueError('Invalid `quantization_dtype`: %r' % quantization_dtype)

  quant_max = np.iinfo(quantization_dtype).max
  scale = (max_val - min_val) / quant_max

  if min_val <= 0 <= max_val:
    quantized_zero_point = (0 - min_val) / scale
    nudged_zero_point = np.round(quantized_zero_point)

    # Solve `0 = nudged_zero_point * scale + nudged_min` for `nudged_min`.
    nudged_min = -nudged_zero_point * scale
    nudged_max = quant_max * scale + nudged_min
  else:
    nudged_min, nudged_max = min_val, max_val

  return scale, nudged_min, nudged_max
