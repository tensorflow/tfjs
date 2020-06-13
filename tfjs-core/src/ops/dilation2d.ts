/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
import {Dilation2D, Dilation2DAttrs, Dilation2DInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor3D, Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {reshape} from './array_ops';
import {op} from './operation';



/**
 * Computes the grayscale dilation over the input `x`.
 *
 * @param x The input tensor, rank 3 or rank 4 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param filter The filter tensor, rank 3, of shape
 *     `[filterHeight, filterWidth, depth]`.
 * @param strides The strides of the sliding window for each dimension of the
 *     input tensor: `[strideHeight, strideWidth]`.
 *     If `strides` is a single number,
 *     then `strideHeight == strideWidth`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1*1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dataFormat Specify the data format of the input and output data.
 *      Defaults to 'NHWC'. Only 'NHWC' is currently supported. With the
 *      default format "NHWC", the data is stored in the order of: [batch,
 *      height, width, channels].
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     for atrous morphological dilation. Defaults to `[1, 1]`. If `dilations`
 *     is a single number, then `dilationHeight == dilationWidth`. If it is
 *     greater than 1, then all values of `strides` must be 1.
 */
/** @doc {heading: 'Operations', subheading: 'Basic math'} */
function dilation2d_<T extends Tensor3D|Tensor4D>(
    x: T|TensorLike, filter: Tensor3D|TensorLike,
    strides: [number, number]|number, pad: 'valid'|'same'|number,
    dataFormat: 'NHWC' = 'NHWC',
    dilations: [number, number]|number = [1, 1]): T {
  const $x = convertToTensor(x, 'x', 'dilation2d');
  const $filter = convertToTensor(filter, 'filter', 'dilation2d');

  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;

  if ($x.rank === 3) {
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
    reshapedTo4D = true;
  }

  util.assert(
      x4D.rank === 4,
      () => `Error in dilation2d: input must be rank 4, but got rank ` +
          `${x4D.rank}.`);
  util.assert(
      $filter.rank === 3,
      () => `Error in dilation2d: filter must be rank 3, but got rank ` +
          `${$filter.rank}.`);
  util.assert(
      dataFormat === 'NHWC',
      () => `Error in dilation2d: Only NHWC is currently supported, ` +
          `but got dataFormat of ${dataFormat}`);

  const inputs: Dilation2DInputs = {x: $x, filter: $filter};
  const attrs: Dilation2DAttrs = {strides, pad, dataFormat, dilations};

  const res = ENGINE.runKernel(
                  Dilation2D, inputs as {} as NamedTensorMap,
                  attrs as {} as NamedAttrMap) as T;

  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as T;
  }

  return res;
}

export const dilation2d = op({dilation2d_});
