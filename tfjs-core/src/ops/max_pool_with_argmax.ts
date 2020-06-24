/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {ENGINE} from '../engine';
import {MaxPoolWithArgmax, MaxPoolWithArgmaxAttrs, MaxPoolWithArgmaxInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {op} from './operation';

/**
 * Computes the 2D max pooling of an image with Argmax index.
 * The indices in argmax are flattened, so that a maximum value at position `[b,
 * y, x, c]` becomes flattened index: `(y * width + x) * channels + c` if
 * include_batch_in_index is False; `((b * height + y) * width + x) * channels
 * +c` if include_batch_in_index is True.
 *
 * The indices returned are always in `[0, height) x [0, width)` before
 * flattening.
 *
 * @param x The input tensor, of rank 4 or rank 3 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param dataFormat An optional string from: "NDHWC", "NCDHW". Defaults to
 *     "NDHWC". Specify the data format of the input and output data. With the
 *     default format "NDHWC", the data is stored in the order of: [batch,
 *     depth, height, width, channels]. Only "NDHWC" is currently supported.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param includeBatchIndex Defaults to False. Whether to include batch
 *    dimension in flattened index of argmax.
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function maxPoolWithArgmax_<T extends Tensor4D>(
    x: T|TensorLike, filterSize: [number, number]|number,
    strides: [number, number]|number, pad: 'valid'|'same'|number,
    includeBatchInIndex = false): NamedTensorMap {
  const $x = convertToTensor(x, 'x', 'maxPoolWithArgmax');

  const inputs: MaxPoolWithArgmaxInputs = {x: $x};
  const attrs:
      MaxPoolWithArgmaxAttrs = {filterSize, strides, pad, includeBatchInIndex};

  const result = ENGINE.runKernel(
                     MaxPoolWithArgmax, inputs as {} as NamedTensorMap,
                     attrs as {} as NamedAttrMap) as Tensor[];

  return {result: result[0], indexes: result[1]};
}

export const maxPoolWithArgmax = op({maxPoolWithArgmax_});
