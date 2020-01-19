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

import {ENGINE} from '../engine';
import {Tensor, Tensor3D, Tensor4D, Tensor5D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';
import {batchToSpaceND, spaceToBatchND} from './array_ops';
import * as conv_util from './conv_util';
import {op} from './operation';

/**
 * Computes the 2D max pooling of an image.
 *
 * @param x The input tensor, of rank 4 or rank 3 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in dilated pooling. Defaults to `[1, 1]`. If `dilations` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
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
function maxPoolImpl_<T extends Tensor3D|Tensor4D>(
    x: T|TensorLike, filterSize: [number, number]|number,
    strides: [number, number]|number, dilations: [number, number]|number,
    pad: 'valid'|'same'|number, dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  const $x = convertToTensor(x, 'x', 'maxPool');

  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
  }
  if (dilations == null) {
    dilations = [1, 1];
  }
  util.assert(
      x4D.rank === 4,
      () => `Error in maxPool: input must be rank 4 but got rank ${x4D.rank}.`);
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in maxPool: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in maxPool: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }
  const convInfo = conv_util.computePool2DInfo(
      x4D.shape, filterSize, strides, dilations, pad, dimRoundingMode);
  if (convInfo.filterWidth === 1 && convInfo.filterHeight === 1 &&
      util.arraysEqual(convInfo.inShape, convInfo.outShape)) {
    return $x.clone();
  }

  const grad = (dy: Tensor4D, saved: Tensor[]) => {
    const [x4D, y] = saved;
    return {
      x: () => maxPoolBackprop(
          dy, x4D as Tensor4D, y as Tensor4D, filterSize, strides, dilations,
          pad)
    };
  };

  const inputsToSave = [x4D];
  const res = ENGINE.runKernelFunc((backend, save) => {
    const y = backend.maxPool(x4D, convInfo);
    save([x4D, y]);
    return y;
  }, {x: x4D}, grad, 'MaxPool', convInfo, inputsToSave);
  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }
  return res as T;
}

/**
 * Computes the 2D max pooling of an image.
 *
 * @param x The input tensor, of rank 4 or rank 3 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
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
  return maxPoolImpl_(x, filterSize, strides, 1, pad, dimRoundingMode);
}

/**
 * Computes the 2D average pooling of an image.
 *
 * @param x The input tensor, of rank 4 or rank 3 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in dilated pooling. Defaults to `[1, 1]`. If `dilations` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
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
function avgPoolImpl_<T extends Tensor3D|Tensor4D>(
    x: T|TensorLike, filterSize: [number, number]|number,
    strides: [number, number]|number, dilations: [number, number]|number,
    pad: 'valid'|'same'|number, dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  const $x = convertToTensor(x, 'x', 'avgPool', 'float32');
  if (dilations == null) {
    dilations = [1, 1];
  }
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in avgPool: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
  }
  util.assert(
      x4D.rank === 4,
      () => `Error in avgPool: x must be rank 4 but got rank ${x4D.rank}.`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in avgPool: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const convInfo = conv_util.computePool2DInfo(
      x4D.shape, filterSize, strides, dilations, pad, dimRoundingMode);
  if (convInfo.filterWidth === 1 && convInfo.filterHeight === 1 &&
      util.arraysEqual(convInfo.inShape, convInfo.outShape)) {
    return $x.clone();
  }

  const grad = (dy: Tensor4D) => {
    return {
      x: () => avgPoolBackprop(dy, x4D, filterSize, strides, dilations, pad)
    };
  };

  let res = ENGINE.runKernelFunc(
      backend => backend.avgPool(x4D, convInfo), {x: x4D}, grad, 'AvgPool',
      convInfo);
  res = res.cast($x.dtype);
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
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
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
  return avgPoolImpl_(x, filterSize, strides, 1, pad, dimRoundingMode);
}

/**
 * Performs an N-D pooling operation
 *
 * @param input The input tensor, of rank 4 or rank 3 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param windowShape The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param poolingType The type of pooling, either 'max' or 'avg'.
 * @param pad The type of padding algorithm:
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *         https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in dilated pooling. Defaults to `[1, 1]`. If `dilationRate` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function pool_<T extends Tensor3D|Tensor4D>(
    input: T|TensorLike, windowShape: [number, number]|number,
    poolingType: 'avg'|'max', pad: 'valid'|'same'|number,
    dilations?: [number, number]|number, strides?: [number, number]|number) {
  if (dilations == null) {
    dilations = [1, 1];
  }
  if (strides == null) {
    strides = 1;
  }
  if (pad === 0) {
    pad = 'valid';
  }
  const $x = convertToTensor(input, 'x', 'maxPool');
  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
  }
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in pool: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
  const convInfo = conv_util.computePool2DInfo(
      x4D.shape, windowShape, strides, dilations, pad);
  const dilation: [number, number] =
      [convInfo.dilationHeight, convInfo.dilationWidth];

  // The following implementation does batchToSpace(pool(spaceToBatch(x)))
  // whenever dilation > 1 since the TF kernels do not support dilation > 1.
  // tslint:disable-next-line:max-line-length
  // https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/nn_ops.py#L1037

  let basePadding: number[][];
  if (pad === 'same') {
    basePadding = withSpaceToBatchBasePaddings(
        [convInfo.filterHeight, convInfo.filterWidth], dilation);
  } else {
    basePadding = [[0, 0], [0, 0]];
  }
  const isDilationOne = dilation[0] === 1 && dilation[1] === 1;
  const [adjustedPadding, adjustedCrops] = requiredSpaceToBatchPaddings(
      [convInfo.inHeight, convInfo.inWidth], dilation, basePadding);
  const convertedPad = isDilationOne ? pad : 'valid';
  const convertedX =
      isDilationOne ? x4D : spaceToBatchND(x4D, dilation, adjustedPadding);
  const forwardOp = poolingType === 'avg' ?
      () => avgPoolImpl_(
          convertedX, windowShape, strides, 1 /* dilation */, convertedPad) :
      () => maxPoolImpl_(
          convertedX, windowShape, strides, 1 /* dilation */, convertedPad);
  const y = forwardOp();
  const res = isDilationOne ? y : batchToSpaceND(y, dilation, adjustedCrops);
  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }
  return res as T;
}

/**
 * Computes the backprop of a 2D max pool.
 *
 * @param dy The dy error, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param input The original input image, of rank 4, of shape
 *     [batchSize, height, width, channels].
 * @param output The original output image, of rank 4, of shape
 *     [batchSize, outHeight, outWidth, channels].
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
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
    strides: [number, number]|number, dilations: [number, number]|number,
    pad: 'valid'|'same'|number,
    dimRoundingMode?: 'floor'|'round'|'ceil'): Tensor4D {
  const $dy = convertToTensor(dy, 'dy', 'maxPoolBackprop');
  const $input = convertToTensor(input, 'input', 'maxPoolBackprop');
  const $output = convertToTensor(output, 'output', 'maxPoolBackprop');
  util.assert(
      $input.rank === $dy.rank,
      () => `Rank of input (${$input.rank}) does not match rank of dy ` +
          `(${$dy.rank})`);
  if (dilations == null) {
    dilations = [1, 1];
  }
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () =>
          'Error in maxPoolBackProp: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);

  util.assert(
      $dy.rank === 4,
      () => `Error in maxPoolBackprop: dy must be rank 4 but got rank ` +
          `${$dy.rank}.`);
  util.assert(
      $input.rank === 4,
      () => `Error in maxPoolBackprop: input must be rank 4 but got rank ` +
          `${$input.rank}.`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in maxPoolBackprop: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const convInfo = conv_util.computePool2DInfo(
      $input.shape, filterSize, strides, dilations, pad, dimRoundingMode);
  const res = ENGINE.runKernelFunc(
      backend => backend.maxPoolBackprop($dy, $input, $output, convInfo),
      {$dy, $input});
  return res;
}

/**
 * Computes the backprop of an 2D avg pool.
 *
 * @param dy The dy error, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param input The input image, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 */
function avgPoolBackprop<T extends Tensor3D|Tensor4D>(
    dy: T|TensorLike, input: T|TensorLike, filterSize: [number, number]|number,
    strides: [number, number]|number, dilations: [number, number]|number,
    pad: 'valid'|'same'|number): T {
  const $dy = convertToTensor(dy, 'dy', 'avgPoolBackprop');
  const $input = convertToTensor(input, 'input', 'avgPoolBackprop');
  util.assert(
      $input.rank === $dy.rank,
      () => `Rank of input (${$input.rank}) does not match rank of dy (${
          $dy.rank})`);
  if (dilations == null) {
    dilations = [1, 1];
  }
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () =>
          'Error in avgPoolBackprop: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);

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
      () => `Error in avgPoolBackprop: dy must be rank 4 but got rank ` +
          `${dy4D.rank}.`);
  util.assert(
      input4D.rank === 4,
      () => `Error in avgPoolBackprop: input must be rank 4 but got rank ` +
          `${input4D.rank}.`);

  const convInfo = conv_util.computePool2DInfo(
      input4D.shape, filterSize, strides, dilations, pad);
  const res = ENGINE.runKernelFunc(
      backend => backend.avgPoolBackprop(dy4D, input4D, convInfo),
      {dy4D, input4D});
  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }
  return res as T;
}

// Helper function to compute crops and paddings for pool with dilation > 1.
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/array_ops.py#L2184
function requiredSpaceToBatchPaddings(
    inputShape: [number, number], blockShape: [number, number],
    basePadding: number[][]) {
  const padStart = basePadding.map(b => b[0]);
  const origPadEnd = basePadding.map(b => b[1]);
  const fullInputShape = inputShape.concat(padStart, origPadEnd);
  const padEndExtra = blockShape.map((b, i) => (b - fullInputShape[i] % b) % b);
  const padEnd = origPadEnd.map((s, i) => s + padEndExtra[i]);
  const paddings = blockShape.map((_, i) => [padStart[i], padEnd[i]]);
  const crops = blockShape.map((_, i) => [0, padEndExtra[i]]);
  return [paddings, crops];
}

// Helper function to compute base paddings for pool with dilation > 1.
// tslint:disable-next-line:max-line-length
// https://github.com/tensorflow/tensorflow/blob/50f6bb67dc98c9b74630b6047aae7a4f8a40fd02/tensorflow/python/ops/nn_ops.py#L524
function withSpaceToBatchBasePaddings(
    filterShape: [number, number], dilation: [number, number]) {
  // Spatial dimensions of the filters and the upsampled filters in which we
  // introduce (rate - 1) zeros between consecutive filter values.
  const dilatedFilterShape = filterShape.map((s, i) => {
    return s + (s - 1) * (dilation[i] - 1);
  });
  const padExtraShape = dilatedFilterShape.map(s => s - 1);

  // When padding is odd, we pad more at end, following the same
  // convention as conv2d.
  const padExtraStart = padExtraShape.map(s => Math.floor(s / 2));
  const padExtraEnd = padExtraShape.map((s, i) => s - padExtraStart[i]);
  return padExtraShape.map((_, i) => {
    return [padExtraStart[i], padExtraEnd[i]];
  });
}

/**
 * Computes the 3D average pooling.
 *
 * ```js
 * const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
 * const result = tf.avgPool3d(x, 2, 1, 'valid');
 * result.print();
 * ```
 *
 * @param x The input tensor, of rank 5 or rank 4 of shape
 *     `[batch, depth, height, width, inChannels]`.
 * @param filterSize The filter size:
 *     `[filterDepth, filterHeight, filterWidth]`.
 *     If `filterSize` is a single number,
 *     then `filterDepth == filterHeight == filterWidth`.
 * @param strides The strides of the pooling:
 *     `[strideDepth, strideHeight, strideWidth]`.
 *     If `strides` is a single number,
 *     then `strideDepth == strideHeight == strideWidth`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1*1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dimRoundingMode The rounding mode used when computing output
 *     dimensions if pad is a number. If none is provided, it will not round
 *     and error if the output is of fractional size.
 * @param dataFormat An optional string from: "NDHWC", "NCDHW". Defaults to
 *     "NDHWC". Specify the data format of the input and output data. With the
 *     default format "NDHWC", the data is stored in the order of: [batch,
 *     depth, height, width, channels]. Only "NDHWC" is currently supported.
 * @param dilations The dilation rates:
 *     `[dilationDepth, dilationHeight, dilationWidth]`
 *     in which we sample input values across the depth, height and width
 *     dimensions in dilated pooling.
 *     Defaults to `[1, 1, 1]`. If `dilations` is a single number,
 *     then `dilationDepth == dilationHeight == dilationWidth`.
 *     If it is greater than 1, then all values of `strides` must be 1.
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function avgPool3d_<T extends Tensor4D|Tensor5D>(
    x: T|TensorLike,
    filterSize: [number, number, number]|number,
    strides: [number, number, number]|number,
    pad: 'valid'|'same'|number,
    dimRoundingMode?: 'floor'|'round'|'ceil',
    dataFormat: 'NDHWC'|'NCDHW' = 'NDHWC',
    dilations?: [number, number, number]|number,
    ): T {
  const $x = convertToTensor(x, 'x', 'avgPool3d', 'float32');

  let x5D = $x as Tensor5D;
  let reshapedTo5D = false;
  if ($x.rank === 4) {
    reshapedTo5D = true;
    x5D = $x.as5D(1, $x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]);
  }

  if (dilations == null) {
    dilations = [1, 1, 1];
  }
  util.assert(
      x5D.rank === 5,
      () => `Error in avgPool3d: x must be rank 5 but got rank ${x5D.rank}.`);
  util.assert(
      dataFormat === 'NDHWC',
      () => `Error in avgPool3d: Only NDHWC is currently supported, ` +
          `but got dataFormat of ${dataFormat}`);
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in avgPool3d: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in avgPool3d: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const convInfo = conv_util.computePool3DInfo(
      x5D.shape, filterSize, strides, dilations, pad, dimRoundingMode,
      dataFormat);

  const grad = (dy: Tensor5D) => {
    return {
      x: () => avgPool3dBackprop(
          dy, x5D, filterSize, strides, dilations, pad, dimRoundingMode)
    };
  };

  let res = ENGINE.runKernelFunc(
      backend => backend.avgPool3d(x5D, convInfo), {x: x5D}, grad);
  res = res.cast(x5D.dtype);
  if (reshapedTo5D) {
    return res.as4D(res.shape[1], res.shape[2], res.shape[3], res.shape[4]) as
        T;
  }

  return res as T;
}

/**
 * Computes the backprop of a 3d avg pool.
 *
 * @param dy The dy error, of rank 5 of shape
 *     [batchSize, depth, height, width, channels].
 * assumed.
 * @param input The original input image, of rank 5 or rank4 of shape
 *     [batchSize, depth, height, width, channels].
 * @param filterSize The filter size:
 *     `[filterDepth, filterHeight, filterWidth]`.
 *     `filterSize` is a single number,
 *     then `filterDepth == filterHeight == filterWidth`.
 * @param strides The strides of the pooling:
 *     `[strideDepth, strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param dilations The dilation rates:
 *     `[dilationDepth, dilationHeight, dilationWidth]`
 *     in which we sample input values across the depth, height and width
 *     dimensions in dilated pooling.
 *     Defaults to `[1, 1, 1]`. If `dilations` is a single number,
 *     then `dilationDepth == dilationHeight == dilationWidth`.
 *     If it is greater than 1, then all values of `strides` must be 1.
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. The
 *     rounding mode used when computing output dimensions if pad is a
 *     number. If none is provided, it will not round and error if the output
 *     is of fractional size.
 */
function avgPool3dBackprop<T extends Tensor4D|Tensor5D>(
    dy: T|TensorLike, input: T|TensorLike,
    filterSize: [number, number, number]|number,
    strides: [number, number, number]|number,
    dilations: [number, number, number]|number, pad: 'valid'|'same'|number,
    dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  const $dy = convertToTensor(dy, 'dy', 'avgPool3dBackprop');
  const $input = convertToTensor(input, 'input', 'avgPool3dBackprop');

  let dy5D = $dy as Tensor5D;
  let input5D = $input as Tensor5D;
  let reshapedTo5D = false;
  if ($input.rank === 4) {
    reshapedTo5D = true;
    dy5D = $dy.as5D(1, $dy.shape[0], $dy.shape[1], $dy.shape[2], $dy.shape[3]);
    input5D = $input.as5D(
        1, $input.shape[0], $input.shape[1], $input.shape[2], $input.shape[3]);
  }

  util.assert(
      dy5D.rank === 5,
      () => `Error in avgPool3dBackprop: dy must be rank 5 but got rank ` +
          `${dy5D.rank}.`);
  util.assert(
      input5D.rank === 5,
      () => `Error in avgPool3dBackprop: input must be rank 5 but got rank ` +
          `${input5D.rank}.`);
  if (dilations == null) {
    dilations = [1, 1, 1];
  }
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in avgPool3dBackprop: Either strides or dilations ' +
          `must be 1. Got strides ${strides} and dilations '${dilations}'`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in maxPool3dBackprop: pad must be an integer when ` +
            `using, dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const convInfo = conv_util.computePool3DInfo(
      input5D.shape, filterSize, strides, dilations, pad, dimRoundingMode);
  const res = ENGINE.runKernelFunc(
      backend => backend.avgPool3dBackprop(dy5D, input5D, convInfo),
      {dy5D, input5D});
  if (reshapedTo5D) {
    return res.as4D(res.shape[1], res.shape[2], res.shape[3], res.shape[4]) as
        T;
  }

  return res as T;
}

/**
 * Computes the 3D max pooling.
 *
 * ```js
 * const x = tf.tensor5d([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 2, 2, 1]);
 * const result = tf.maxPool3d(x, 2, 1, 'valid');
 * result.print();
 * ```
 *
 * @param x The input tensor, of rank 5 or rank 4 of shape
 *     `[batch, depth, height, width, inChannels]`.
 * @param filterSize The filter size:
 *     `[filterDepth, filterHeight, filterWidth]`.
 *     If `filterSize` is a single number,
 *     then `filterDepth == filterHeight == filterWidth`.
 * @param strides The strides of the pooling:
 *     `[strideDepth, strideHeight, strideWidth]`.
 *     If `strides` is a single number,
 *     then `strideDepth == strideHeight == strideWidth`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1*1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dimRoundingMode The rounding mode used when computing output
 *     dimensions if pad is a number. If none is provided, it will not round
 *     and error if the output is of fractional size.
 * @param dataFormat An optional string from: "NDHWC", "NCDHW". Defaults to
 *     "NDHWC". Specify the data format of the input and output data. With the
 *     default format "NDHWC", the data is stored in the order of: [batch,
 *     depth, height, width, channels]. Only "NDHWC" is currently supported.
 * @param dilations The dilation rates:
 *     `[dilationDepth, dilationHeight, dilationWidth]`
 *     in which we sample input values across the depth, height and width
 *     dimensions in dilated pooling.
 *     Defaults to `[1, 1, 1]`. If `dilations` is a single number,
 *     then `dilationDepth == dilationHeight == dilationWidth`.
 *     If it is greater than 1, then all values of `strides` must be 1.
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function maxPool3d_<T extends Tensor4D|Tensor5D>(
    x: T|TensorLike, filterSize: [number, number, number]|number,
    strides: [number, number, number]|number, pad: 'valid'|'same'|number,
    dimRoundingMode?: 'floor'|'round'|'ceil',
    dataFormat: 'NDHWC'|'NCDHW' = 'NDHWC',
    dilations?: [number, number, number]|number): T {
  const $x = convertToTensor(x, 'x', 'maxPool3d');

  let x5D = $x as Tensor5D;
  let reshapedTo5D = false;
  if ($x.rank === 4) {
    reshapedTo5D = true;
    x5D = $x.as5D(1, $x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]);
  }

  if (dilations == null) {
    dilations = [1, 1, 1];
  }
  util.assert(
      x5D.rank === 5,
      () => `Error in maxPool3d: x must be rank 5 but got rank ${x5D.rank}.`);
  util.assert(
      dataFormat === 'NDHWC',
      () => `Error in maxPool3d: Only NDHWC is currently supported, ` +
          `but got dataFormat of ${dataFormat}`);
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in maxPool3d: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in maxPool3d: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const convInfo = conv_util.computePool3DInfo(
      x5D.shape, filterSize, strides, dilations, pad, dimRoundingMode,
      dataFormat);

  const grad = (dy: Tensor5D, saved: Tensor[]) => {
    const [x5D, y] = saved;
    return {
      x: () => maxPool3dBackprop(
          dy, x5D as Tensor5D, y as Tensor5D, filterSize, strides, dilations,
          pad, dimRoundingMode)
    };
  };

  const res = ENGINE.runKernelFunc((backend, save) => {
    const y = backend.maxPool3d(x5D, convInfo);
    save([x5D, y]);
    return y;
  }, {x: x5D}, grad);
  if (reshapedTo5D) {
    return res.as4D(res.shape[1], res.shape[2], res.shape[3], res.shape[4]) as
        T;
  }

  return res as T;
}

/**
 * Computes the backprop of a 3d max pool.
 *
 * @param dy The dy error, of rank 5 of shape
 *     [batchSize, depth, height, width, channels].
 * assumed.
 * @param input The original input image, of rank 5 or rank 4 of shape
 *     [batchSize, depth, height, width, channels].
 * @param output The original output image, of rank 5 of shape
 *     [batchSize, outDepth, outHeight, outWidth, channels].
 * @param filterSize The filter size:
 *     `[filterDepth, filterHeight, filterWidth]`.
 *     `filterSize` is a single number,
 *     then `filterDepth == filterHeight == filterWidth`.
 * @param strides The strides of the pooling:
 *     `[strideDepth, strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param dilations The dilation rates:
 *     `[dilationDepth, dilationHeight, dilationWidth]`
 *     in which we sample input values across the depth, height and width
 *     dimensions in dilated pooling.
 *     Defaults to `[1, 1, 1]`. If `dilations` is a single number,
 *     then `dilationDepth == dilationHeight == dilationWidth`.
 *     If it is greater than 1, then all values of `strides` must be 1.
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. The
 *     rounding mode used when computing output dimensions if pad is a
 *     number. If none is provided, it will not round and error if the output
 *     is of fractional size.
 */
function maxPool3dBackprop<T extends Tensor4D|Tensor5D>(
    dy: T|TensorLike, input: T|TensorLike, output: T|TensorLike,
    filterSize: [number, number, number]|number,
    strides: [number, number, number]|number,
    dilations: [number, number, number]|number, pad: 'valid'|'same'|number,
    dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  const $dy = convertToTensor(dy, 'dy', 'maxPool3dBackprop');
  const $input = convertToTensor(input, 'input', 'maxPool3dBackprop');
  const $output = convertToTensor(output, 'output', 'maxPool3dBackprop');

  let dy5D = $dy as Tensor5D;
  let input5D = $input as Tensor5D;
  let output5D = $output as Tensor5D;
  let reshapedTo5D = false;
  if ($input.rank === 4) {
    reshapedTo5D = true;
    dy5D = $dy.as5D(1, $dy.shape[0], $dy.shape[1], $dy.shape[2], $dy.shape[3]);
    input5D = $input.as5D(
        1, $input.shape[0], $input.shape[1], $input.shape[2], $input.shape[3]);
    output5D = $output.as5D(
        1, $output.shape[0], $output.shape[1], $output.shape[2],
        $output.shape[3]);
  }

  util.assert(
      dy5D.rank === 5,
      () => `Error in maxPool3dBackprop: dy must be rank 5 but got rank ` +
          `${dy5D.rank}.`);
  util.assert(
      input5D.rank === 5,
      () => `Error in maxPool3dBackprop: input must be rank 5 but got rank ` +
          `${input5D.rank}.`);
  util.assert(
      output5D.rank === 5,
      () => `Error in maxPool3dBackprop: output must be rank 5 but got rank ` +
          `${output5D.rank}.`);
  if (dilations == null) {
    dilations = [1, 1, 1];
  }
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in maxPool3dBackprop: Either strides or dilations ' +
          `must be 1. Got strides ${strides} and dilations '${dilations}'`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in maxPool3dBackprop: pad must be an integer when ` +
            `using, dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const convInfo = conv_util.computePool3DInfo(
      input5D.shape, filterSize, strides, dilations, pad, dimRoundingMode);
  const res = ENGINE.runKernelFunc(
      backend => backend.maxPool3dBackprop(dy5D, input5D, output5D, convInfo),
      {dy5D, input5D});

  if (reshapedTo5D) {
    return res.as4D(res.shape[1], res.shape[2], res.shape[3], res.shape[4]) as
        T;
  }

  return res as T;
}

export const maxPool = op({maxPool_});
export const avgPool = op({avgPool_});
export const pool = op({pool_});
export const maxPool3d = op({maxPool3d_});
export const avgPool3d = op({avgPool3d_});
