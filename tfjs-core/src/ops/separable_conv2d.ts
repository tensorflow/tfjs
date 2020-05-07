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
import {Tensor3D, Tensor4D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {conv2d} from './conv2d';
import {depthwiseConv2d} from './depthwise_conv2d';
import {op} from './operation';

/**
 * 2-D convolution with separable filters.
 *
 * Performs a depthwise convolution that acts separately on channels followed
 * by a pointwise convolution that mixes channels. Note that this is
 * separability between dimensions [1, 2] and 3, not spatial separability
 * between dimensions 1 and 2.
 *
 * See
 * [https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d](
 *     https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d)
 * for more details.
 *
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param depthwiseFilter The depthwise filter tensor, rank 4, of shape
 *     `[filterHeight, filterWidth, inChannels, channelMultiplier]`. This is
 *     the filter used in the first step.
 * @param pointwiseFilter The pointwise filter tensor, rank 4, of shape
 *     `[1, 1, inChannels * channelMultiplier, outChannels]`. This is
 *     the filter used in the second step.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`. If strides is a single number, then `strideHeight ==
 * strideWidth`.
 * @param pad The type of padding algorithm.
 *   - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *   - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in atrous convolution. Defaults to `[1, 1]`. If `rate` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels]. Only "NHWC" is currently supported.
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function separableConv2d_<T extends Tensor3D|Tensor4D>(
    x: T|TensorLike, depthwiseFilter: Tensor4D|TensorLike,
    pointwiseFilter: Tensor4D|TensorLike, strides: [number, number]|number,
    pad: 'valid'|'same', dilation: [number, number]|number = [1, 1],
    dataFormat: 'NHWC'|'NCHW' = 'NHWC'): T {
  const $x = convertToTensor(x, 'x', 'separableConv2d');
  const $depthwiseFilter =
      convertToTensor(depthwiseFilter, 'depthwiseFilter', 'separableConv2d');
  const $pointwiseFilter =
      convertToTensor(pointwiseFilter, 'pointwiseFilter', 'separableConv2d');

  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
  }

  if (dataFormat === 'NCHW') {
    throw new Error(
        'separableConv2d currently does not support dataFormat NCHW; only ' +
        'NHWC is supported');
  }

  util.assert(
      x4D.rank === 4,
      () => `Error in separableConv2d: input must be rank 4, but got ` +
          `rank ${x4D.rank}.`);
  util.assert(
      $depthwiseFilter.rank === 4,
      () => `Error in separableConv2d: depthwise filter must be rank 4, but ` +
          `got rank ${$depthwiseFilter.rank}.`);
  util.assert(
      $pointwiseFilter.rank === 4,
      () => `Error in separableConv2d: pointwise filter must be rank 4, but ` +
          `got rank ${$depthwiseFilter.rank}.`);
  util.assert(
      $pointwiseFilter.shape[0] === 1,
      () =>
          `Error in separableConv2d: the first dimension of pointwise filter ` +
          ` must be 1, but got ${$pointwiseFilter.shape[0]}.`);
  util.assert(
      $pointwiseFilter.shape[1] === 1,
      () => `Error in separableConv2d: the second dimension of pointwise ` +
          `filter must be 1, but got ${$pointwiseFilter.shape[1]}.`);

  const inChannels = $depthwiseFilter.shape[2];
  const channelMultiplier = $depthwiseFilter.shape[3];
  util.assert(
      $pointwiseFilter.shape[2] === inChannels * channelMultiplier,
      () =>
          `Error in separableConv2d: the third dimension of pointwise filter ` +
          `must be ${inChannels * channelMultiplier}, ` +
          `but got ${$pointwiseFilter.shape[2]}.`);

  const depthwise = depthwiseConv2d(
      x4D, $depthwiseFilter, strides, pad, dataFormat, dilation);
  const pointwiseStride = 1;
  const res =
      conv2d(depthwise, $pointwiseFilter, pointwiseStride, 'valid', dataFormat);

  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }
  return res as T;
}

export const separableConv2d = op({separableConv2d_});
