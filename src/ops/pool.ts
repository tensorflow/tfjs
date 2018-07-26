/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {ENV} from '../environment';
import {Tensor, Tensor3D, Tensor4D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';
import * as conv_util from './conv_util';
import {op} from './operation';

/**
 * Computes the 2D max pooling of an image.
 *
 * @param x The input tensor, of rank 4 or rank 3 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param filterSize The filter size, a tuple `[filterHeight, filterWidth]`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dimRoundingMode The rounding mode used when computing output
 *     dimensions if pad is a number. If none is provided, it will not round
 *     and error if the output is of fractional size.
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function maxPool_<T extends Tensor3D|Tensor4D>(
    x: T|TensorLike, filterSize: [number, number]|number,
    strides: [number, number]|number, pad: 'valid'|'same'|number,
    dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  const $x = convertToTensor(x, 'x', 'maxPool');

  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
  }
  util.assert(
      x4D.rank === 4,
      `Error in maxPool: input must be rank 4 but got rank ${x4D.rank}.`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        `Error in maxPool: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }
  const convInfo = conv_util.computePool2DInfo(
      x4D.shape, filterSize, strides, pad, dimRoundingMode);

  const grad = (dy: Tensor4D, saved: Tensor[]) => {
    const [y4D] = saved;
    return {
      x: () =>
          maxPoolBackprop(dy, x4D, y4D as Tensor4D, filterSize, strides, pad)
    };
  };

  const res = ENV.engine.runKernel(
      (backend, save) => save(backend.maxPool(x4D, convInfo)), {x: x4D}, grad);
  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }
  return res as T;
}

/**
 * Computes the 2D average pooling of an image.
 *
 * @param x The input tensor, of rank 4 or rank 3 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param filterSize The filter size, a tuple `[filterHeight, filterWidth]`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`.
 * @param pad The type of padding algorithm:
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *         https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dimRoundingMode The rounding mode used when computing output
 *     dimensions if pad is a number. If none is provided, it will not round
 *     and error if the output is of fractional size.
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function avgPool_<T extends Tensor3D|Tensor4D>(
    x: T|TensorLike, filterSize: [number, number]|number,
    strides: [number, number]|number, pad: 'valid'|'same'|number,
    dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  const $x = convertToTensor(x, 'x', 'avgPool');
  util.assert(
      $x.dtype === 'float32', 'The input dtype to avgPool must be float32');
  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
  }
  util.assert(
      x4D.rank === 4,
      `Error in avgPool: x must be rank 4 but got rank ${x4D.rank}.`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        `Error in avgPool: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const convInfo =
      conv_util.computePool2DInfo(x4D.shape, filterSize, strides, pad);

  const grad = (dy: Tensor4D) => {
    return {x: () => avgPoolBackprop(dy, x4D, filterSize, strides, pad)};
  };
  let res = ENV.engine.runKernel(
      backend => backend.avgPool(x4D, convInfo), {x: x4D}, grad);
  res = res.cast($x.dtype);
  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }
  return res as T;
}

/**
 * Computes the backprop of a max pool.
 *
 * @param dy The dy error, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param input The original input image, of rank 4, of shape
 *     [batchSize, height, width, channels].
 * @param output The original output image, of rank 4, of shape
 *     [batchSize, outHeight, outWidth, channels].
 * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
 * @param strides The strides of the pooling: [strideHeight, strideWidth].
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. The
 *     rounding mode used when computing output dimensions if pad is a
 *     number. If none is provided, it will not round and error if the output
 *     is of fractional size.
 */
function maxPoolBackprop(
    dy: Tensor4D|TensorLike, input: Tensor4D|TensorLike,
    output: Tensor4D|TensorLike, filterSize: [number, number]|number,
    strides: [number, number]|number, pad: 'valid'|'same'|number,
    dimRoundingMode?: 'floor'|'round'|'ceil'): Tensor4D {
  const $dy = convertToTensor(dy, 'dy', 'maxPoolBackprop');
  const $input = convertToTensor(input, 'input', 'maxPoolBackprop');
  const $output = convertToTensor(output, 'output', 'maxPoolBackprop');
  util.assert(
      $input.rank === $dy.rank,
      `Rank of input (${$input.rank}) does not match rank of dy (${$dy.rank})`);

  util.assert(
      $dy.rank === 4,
      `Error in maxPoolBackprop: dy must be rank 4 but got rank ` +
          `${$dy.rank}.`);
  util.assert(
      $input.rank === 4,
      `Error in maxPoolBackprop: input must be rank 4 but got rank ` +
          `${$input.rank}.`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        `Error in maxPoolBackprop: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const convInfo = conv_util.computePool2DInfo(
      $input.shape, filterSize, strides, pad, dimRoundingMode);
  const res = ENV.engine.runKernel(
      backend => backend.maxPoolBackprop($dy, $input, $output, convInfo),
      {$dy, $input});
  return res;
}

/**
 * Computes the backprop of an avg pool.
 *
 * @param dy The dy error, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param input The input image, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param filterSize The filter size, a tuple [filterHeight, filterWidth].
 * @param strides The strides of the pooling: [strideHeight, strideWidth].
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 */
function avgPoolBackprop<T extends Tensor3D|Tensor4D>(
    dy: T|TensorLike, input: T|TensorLike, filterSize: [number, number]|number,
    strides: [number, number]|number, pad: 'valid'|'same'|number): T {
  const $dy = convertToTensor(dy, 'dy', 'avgPoolBackprop');
  const $input = convertToTensor(input, 'input', 'avgPoolBackprop');
  util.assert(
      $input.rank === $dy.rank,
      `Rank of input (${$input.rank}) does not match rank of dy (${$dy.rank})`);

  let input4D = $input as Tensor4D;
  let dy4D = $dy as Tensor4D;
  let reshapedTo4D = false;
  if ($input.rank === 3) {
    reshapedTo4D = true;
    input4D = $input.as4D(1, $input.shape[0], $input.shape[1], $input.shape[2]);
    dy4D = $dy.as4D(1, $dy.shape[0], $dy.shape[1], $dy.shape[2]);
  }

  util.assert(
      dy4D.rank === 4,
      `Error in avgPoolBackprop: dy must be rank 4 but got rank ` +
          `${dy4D.rank}.`);
  util.assert(
      input4D.rank === 4,
      `Error in avgPoolBackprop: input must be rank 4 but got rank ` +
          `${input4D.rank}.`);

  const convInfo =
      conv_util.computePool2DInfo(input4D.shape, filterSize, strides, pad);
  const res = ENV.engine.runKernel(
      backend => backend.avgPoolBackprop(dy4D, input4D, convInfo),
      {dy4D, input4D});
  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }
  return res as T;
}

export const maxPool = op({maxPool_});
export const avgPool = op({avgPool_});
