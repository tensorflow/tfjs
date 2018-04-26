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

import numpy as np

QUANTIZATION_BYTES_TO_DTYPES = {1: np.uint8, 2: np.uint16}


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
    quantization_dtype: A numpy dtype to quantize weights to. Only np.uint8 and
      np.uint16 are supported.

  Returns:
    quantized_data: The quantized weights as a numpy array with dtype
      `quantization_dtype`.
    scale: The linearly scaling constant used for quantization.
    min_val: The minimum value of the linear range.
  Raises:
    ValueError: if `quantization_dtype` is not a valid type.
  """
  if quantization_dtype not in QUANTIZATION_BYTES_TO_DTYPES.values():
    raise ValueError('Invalid `quantization_dtype`: %r' % quantization_dtype)

  # Compute the min and max for the group.
  min_val = data.min().astype(np.float64)
  max_val = data.max().astype(np.float64)
  if min_val == max_val:
    # If there is only a single value, we can represent everything as zeros.
    quantized_data = np.zeros_like(data, dtype=quantization_dtype)
    scale = 1.0
  else:
    # Quantize data.
    scale, min_val, max_val = _get_quantization_range(
        min_val, max_val, quantization_dtype)
    quantized_data = np.round(
        (data.clip(min_val, max_val) - min_val) / scale).astype(
            quantization_dtype)

  return quantized_data, scale, min_val


def dequantize_weights(
    quantized_data, scale, min_val, original_dtype=np.float32):
  return np.round(quantized_data * scale + min_val).astype(original_dtype)

def _get_quantization_range(min_val, max_val, quantization_dtype):
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
  if quantization_dtype not in QUANTIZATION_BYTES_TO_DTYPES.values():
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
