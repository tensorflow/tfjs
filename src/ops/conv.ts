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
import {Tensor, Tensor2D, Tensor3D, Tensor4D, Tensor5D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';
import * as conv_util from './conv_util';
import {op} from './operation';

/**
 * Computes a 1D convolution over the input x.
 *
 * @param x The input tensor, of rank 3 or rank 2, of shape
 *     `[batch, width, inChannels]`. If rank 2, batch of 1 is assumed.
 * @param filter The filter, rank 3, of shape
 *     `[filterWidth, inDepth, outDepth]`.
 * @param stride The number of entries by which the filter is moved right at
 *     each step.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dataFormat An optional string from "NWC", "NCW". Defaults to "NWC",
 *     the data is stored in the order of [batch, in_width, in_channels]. Only
 *     "NWC" is currently supported.
 * @param dilation The dilation rate in which we sample input values in
 *     atrous convolution. Defaults to `1`. If it is greater than 1, then
 *     stride must be `1`.
 * @param dimRoundingMode The rounding mode used when computing output
 *     dimensions if pad is a number. If none is provided, it will not round
 *     and error if the output is of fractional size.
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function conv1d_<T extends Tensor2D|Tensor3D>(
    x: T|TensorLike, filter: Tensor3D|TensorLike, stride: number,
    pad: 'valid'|'same'|number, dataFormat: 'NWC'|'NCW' = 'NWC', dilation = 1,
    dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  const $x = convertToTensor(x, 'x', 'conv1d');
  const $filter = convertToTensor(filter, 'filter', 'conv1d');

  let x3D = $x as Tensor3D;
  let reshapedTo3D = false;
  if ($x.rank === 2) {
    reshapedTo3D = true;
    x3D = $x.as3D(1, $x.shape[0], $x.shape[1]);
  }

  util.assert(
      x3D.rank === 3,
      () => `Error in conv1d: input must be rank 3, but got rank ${x3D.rank}.`);
  util.assert(
      $filter.rank === 3,
      () => `Error in conv1d: filter must be rank 3, but got rank ` +
          `${$filter.rank}.`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in conv1d: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  util.assert(
      x3D.shape[2] === $filter.shape[1],
      () => `Error in conv1d: depth of input (${x3D.shape[2]}) must match ` +
          `input depth for filter ${$filter.shape[1]}.`);
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(stride, dilation),
      () => 'Error in conv1D: Either stride or dilation must be 1. ' +
          `Got stride ${stride} and dilation '${dilation}'`);
  util.assert(
      dataFormat === 'NWC',
      () => `Error in conv1d: got dataFormat of ${
          dataFormat} but only NWC is currently supported.`);

  const filter4D =
      $filter.as4D(1, $filter.shape[0], $filter.shape[1], $filter.shape[2]);
  const input4D = x3D.as4D(x3D.shape[0], 1, x3D.shape[1], x3D.shape[2]);
  const strides: [number, number] = [1, stride];
  const dilations: [number, number] = [1, dilation];

  const conv2dDataFormat = 'NHWC';

  const res = conv2d(
      input4D, filter4D, strides, pad, conv2dDataFormat, dilations,
      dimRoundingMode);

  if (reshapedTo3D) {
    return res.as2D(res.shape[2], res.shape[3]) as T;
  }
  return res.as3D(res.shape[0], res.shape[2], res.shape[3]) as T;
}

/**
 * Computes a 2D convolution over the input x.
 *
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter, rank 4, of shape
 *     `[filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels]. Only "NHWC" is currently supported.
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in atrous convolution. Defaults to `[1, 1]`. If `dilations` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param dimRoundingMode The rounding mode used when computing output
 *     dimensions if pad is a number. If none is provided, it will not round
 *     and error if the output is of fractional size.
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function conv2d_<T extends Tensor3D|Tensor4D>(
    x: T|TensorLike, filter: Tensor4D|TensorLike,
    strides: [number, number]|number, pad: 'valid'|'same'|number,
    dataFormat: 'NHWC'|'NCHW' = 'NHWC',
    dilations: [number, number]|number = [1, 1],
    dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  const $x = convertToTensor(x, 'x', 'conv2d');
  const $filter = convertToTensor(filter, 'filter', 'conv2d');

  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;

  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
  }
  util.assert(
      x4D.rank === 4,
      () => `Error in conv2d: input must be rank 4, but got rank ${x4D.rank}.`);
  util.assert(
      $filter.rank === 4,
      () => `Error in conv2d: filter must be rank 4, but got rank ` +
          `${$filter.rank}.`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in conv2d: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  util.assert(
      x4D.shape[3] === $filter.shape[2],
      () => `Error in conv2d: depth of input (${x4D.shape[3]}) must match ` +
          `input depth for filter ${$filter.shape[2]}.`);
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in conv2D: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
  util.assert(
      dataFormat === 'NHWC',
      () => `Error in conv2d: got dataFormat of ${
          dataFormat} but only NHWC is currently supported.`);

  const convInfo = conv_util.computeConv2DInfo(
      x4D.shape, $filter.shape, strides, dilations, pad, dimRoundingMode);

  const grad = (dy: Tensor4D, saved: Tensor[]) => {
    const [$filter, x4D] = saved as [Tensor4D, Tensor4D];
    util.assert(
        conv_util.tupleValuesAreOne(dilations),
        () => 'Error in gradient of conv2D: dilation rates greater than 1 ' +
            `are not yet supported in gradients. Got dilations '${dilations}'`);

    return {
      x: () => conv2dDerInput_(x4D.shape, dy, $filter, strides, pad),
      $filter: () => conv2dDerFilter_(x4D, dy, $filter.shape, strides, pad)
    };
  };

  const res = ENGINE.runKernel((backend, save) => {
    const res = backend.conv2d(x4D, $filter, convInfo);
    save([$filter, x4D]);

    return res;
  }, {x: x4D, $filter}, grad);

  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }
  return res as T;
}

/**
 * Computes the derivative of the input of a 2D convolution.
 *
 * @param xShape The shape of the input: [batch, height, width, inDepth].
 * If length of 3, batch of 1 is assumed.
 * @param dy The derivative of the output, of rank 4 or rank 3 of shape
 *   `[batch, outHeight, outWidth, outDepth]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter, rank 4, of shape
 *     `[filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm used:
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 * @param dimRoundingMode The rounding mode used when computing output
 *     dimensions if pad is a number. If none is provided, it will not round
 *     and error if the output is of fractional size.
 */
function conv2dDerInput_<T extends Tensor3D|Tensor4D>(
    xShape: [number, number, number, number]|[number, number, number], dy: T,
    filter: Tensor4D, strides: [number, number]|number,
    pad: 'valid'|'same'|number, dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  util.assert(
      xShape.length === dy.rank,
      () => `Length of inShape ` +
          `(${xShape.length}) and rank of dy (${dy.rank}) must match`);

  let xShape4D = xShape as [number, number, number, number];
  let dy4D = dy as Tensor4D;
  let reshapedTo4D = false;
  if (dy.rank === 3) {
    reshapedTo4D = true;
    dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
    xShape4D = [1, xShape[0], xShape[1], xShape[2]];
  }

  const inDepth = xShape4D[3];
  const outDepth = dy4D.shape[3];
  util.assert(
      xShape4D.length === 4,
      () =>
          `Error in conv2dDerInput: inShape must be length 4, but got length ` +
          `${xShape4D.length}.`);
  util.assert(
      dy4D.rank === 4,
      () => `Error in conv2dDerInput: dy must be rank 4, but got ` +
          `rank ${dy4D.rank}`);
  util.assert(
      filter.rank === 4,
      () => `Error in conv2dDerInput: filter must be rank 4, but got ` +
          `rank ${filter.rank}`);
  util.assert(
      inDepth === filter.shape[2],
      () => `Error in conv2dDerInput: depth of input (${inDepth}) must ` +
          `match input depth for filter ${filter.shape[2]}.`);
  util.assert(
      outDepth === filter.shape[3],
      () => `Error in conv2dDerInput: depth of output (${outDepth}) must ` +
          `match output depth for filter ${filter.shape[3]}.`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in conv2dDerInput: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const dilations = 1;

  const grad = (ddx: Tensor4D, saved: Tensor[]) => {
    const dataFormat = 'NHWC';
    const [filter, dy4D] = saved;
    return {
      dy4D: () => conv2d(
          ddx, filter as Tensor4D, strides, pad, dataFormat, dilations,
          dimRoundingMode),
      filter: () => conv2dDerFilter(
          ddx, dy4D as Tensor4D, (filter as Tensor4D).shape, strides, pad,
          dimRoundingMode)
    };
  };

  const convInfo = conv_util.computeConv2DInfo(
      xShape4D, filter.shape, strides, dilations, pad, dimRoundingMode);
  const res = ENGINE.runKernel((backend, save) => {
    const res = backend.conv2dDerInput(dy4D, filter, convInfo);
    save([filter, dy4D]);
    return res;
  }, {dy4D, filter}, grad);
  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }
  return res as T;
}

/**
 * Computes the derivative of the filter of a 2D convolution.
 *
 * @param x The input tensor, of rank 4 or rank 3 of shape
 *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
 * @param dy The dy image, of rank 4 or rank 3, of shape
 *     [batch, height, width, outDepth]. If rank 3, batch of 1 is assumed.
 * @param filterShape The shape of the filter, length 4,
 *     [filterHeight, filterWidth, inDepth, outDepth].
 * @param strides The strides of the convolution: [strideHeight,
 * strideWidth].
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. The
 *     rounding mode used when computing output dimensions if pad is a
 *     number. If none is provided, it will not round and error if the output
 *     is of fractional size.
 */
function conv2dDerFilter_<T extends Tensor3D|Tensor4D>(
    x: T, dy: T, filterShape: [number, number, number, number],
    strides: [number, number]|number, pad: 'valid'|'same'|number,
    dimRoundingMode?: 'floor'|'round'|'ceil'): Tensor4D {
  let x4D = x as Tensor4D;
  if (x.rank === 3) {
    x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
  }
  let dy4D = dy as Tensor4D;
  if (dy4D.rank === 3) {
    dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
  }
  util.assert(
      x4D.rank === 4,
      () => `Error in conv2dDerFilter: input must be rank 4, but got shape ` +
          `${x4D.shape}.`);
  util.assert(
      dy4D.rank === 4,
      () => `Error in conv2dDerFilter: dy must be rank 4, but got shape ` +
          `${dy4D.shape}.`);
  util.assert(
      filterShape.length === 4,
      () => `Error in conv2dDerFilter: filterShape must be length 4, but got ` +
          `${filterShape}.`);
  util.assert(
      x4D.shape[3] === filterShape[2],
      () => `Error in conv2dDerFilter: depth of input ${x4D.shape[3]}) must ` +
          `match input depth in filter (${filterShape[2]}.`);
  util.assert(
      dy4D.shape[3] === filterShape[3],
      () => `Error in conv2dDerFilter: depth of dy (${dy4D.shape[3]}) must ` +
          `match output depth for filter (${filterShape[3]}).`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in conv2dDerFilter: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const dilations = 1;

  const convInfo = conv_util.computeConv2DInfo(
      x4D.shape, filterShape, strides, dilations, pad, dimRoundingMode);
  return ENGINE.runKernel(
      backend => backend.conv2dDerFilter(x4D, dy4D, convInfo), {x4D, dy4D});
}

/**
 * Computes the transposed 2D convolution of an image, also known as a
 * deconvolution.
 *
 * @param x The input image, of rank 4 or rank 3, of shape
 *   `[batch, height, width, inDepth]`. If rank 3, batch of 1 is assumed.
 * @param filter The filter, rank 4, of shape
 *     `[filterHeight, filterWidth, outDepth, inDepth]`.
 *     `inDepth` must match `inDepth` in `x`.
 * @param outputShape Output shape, of rank 4 or rank 3:
 *     `[batch, height, width, outDepth]`. If rank 3, batch of 1 is assumed.
 * @param strides The strides of the original convolution:
 *     `[strideHeight, strideWidth]`.
 * @param pad  The type of padding algorithm used in the non-transpose version
 *    of the op.
 * @param dimRoundingMode The rounding mode used when computing output
 *    dimensions if pad is a number. If none is provided, it will not round
 *    and error if the output is of fractional size.
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function conv2dTranspose_<T extends Tensor3D|Tensor4D>(
    x: T|TensorLike, filter: Tensor4D|TensorLike,
    outputShape: [number, number, number, number]|[number, number, number],
    strides: [number, number]|number, pad: 'valid'|'same'|number,
    dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  const $x = convertToTensor(x, 'x', 'conv2dTranspose');
  const $filter = convertToTensor(filter, 'filter', 'conv2dTranspose');

  return conv2dDerInput_(
      outputShape, $x, $filter, strides, pad, dimRoundingMode);
}

/**
 * Depthwise 2D convolution.
 *
 * Given a 4D `input` array and a `filter` array of shape
 * `[filterHeight, filterWidth, inChannels, channelMultiplier]` containing
 * `inChannels` convolutional filters of depth 1, this op applies a
 * different filter to each input channel (expanding from 1 channel to
 * `channelMultiplier` channels for each), then concatenates the results
 * together. The output has `inChannels * channelMultiplier` channels.
 *
 * See
 * [https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d](
 *     https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)
 * for more details.
 *
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter tensor, rank 4, of shape
 *     `[filterHeight, filterWidth, inChannels, channelMultiplier]`.
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
 * @param dimRoundingMode The rounding mode used when computing output
 *     dimensions if pad is a number. If none is provided, it will not round
 *     and error if the output is of fractional size.
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function depthwiseConv2d_<T extends Tensor3D|Tensor4D>(
    x: T|TensorLike, filter: Tensor4D|TensorLike,
    strides: [number, number]|number, pad: 'valid'|'same'|number,
    dataFormat: 'NHWC'|'NCHW' = 'NHWC',
    dilations: [number, number]|number = [1, 1],
    dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  const $x = convertToTensor(x, 'x', 'depthwiseConv2d');
  const $filter = convertToTensor(filter, 'filter', 'depthwiseConv2d');

  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
  }
  util.assert(
      x4D.rank === 4,
      () => `Error in depthwiseConv2d: input must be rank 4, but got ` +
          `rank ${x4D.rank}.`);
  util.assert(
      $filter.rank === 4,
      () => `Error in depthwiseConv2d: filter must be rank 4, but got rank ` +
          `${$filter.rank}.`);
  util.assert(
      x4D.shape[3] === $filter.shape[2],
      () => `Error in depthwiseConv2d: number of input channels ` +
          `(${x4D.shape[3]}) must match the inChannels dimension in ` +
          `filter ${$filter.shape[2]}.`);
  if (dilations == null) {
    dilations = [1, 1];
  }
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () =>
          'Error in depthwiseConv2d: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);

  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in depthwiseConv2d: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const convInfo = conv_util.computeConv2DInfo(
      x4D.shape, $filter.shape, strides, dilations, pad, dimRoundingMode,
      true /* depthwise */);

  const grad = (dy: Tensor4D, saved: Tensor[]) => {
    util.assert(
        conv_util.tupleValuesAreOne(dilations),
        () => 'Error in gradient of depthwiseConv2d: dilation rates ' +
            `greater than 1 are not yet supported. Got dilations ` +
            `'${dilations}'`);
    const [x4D, $filter] = saved;
    return {
      x: () => depthwiseConv2dDerInput(
          (x4D as Tensor4D).shape, dy, $filter as Tensor4D, convInfo),
      $filter: () => depthwiseConv2dDerFilter(
          x4D as Tensor4D, dy, ($filter as Tensor4D).shape, convInfo),
    };
  };

  const res = ENGINE.runKernel((backend, save) => {
    const res = backend.depthwiseConv2D(x4D, $filter, convInfo);
    save([x4D, $filter]);
    return res;
  }, {x: x4D, $filter}, grad);
  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }
  return res as T;
}

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

function parseTupleParam(
    param: number|[number, number]|[number, number, number]):
    [number, number, number] {
  if (typeof param === 'number') {
    return [param, param, param];
  }
  if (param.length === 2) {
    return [param[0], param[1], 1];
  }
  return param;
}

function tupleValuesAreOne(
    param: number|[number, number]|[number, number, number]): boolean {
  const [dimA, dimB, dimC] = parseTupleParam(param);
  return dimA === 1 && dimB === 1 && dimC === 1;
}

function eitherStridesOrDilationsAreOne(
    strides: number|[number, number]|[number, number, number],
    dilations: number|[number, number]|[number, number, number]): boolean {
  return tupleValuesAreOne(strides) || tupleValuesAreOne(dilations);
}

function depthwiseConv2dDerInput<T extends Tensor3D|Tensor4D>(
    xShape: [number, number, number, number]|[number, number, number], dy: T,
    filter: Tensor4D, convInfo: conv_util.Conv2DInfo): T {
  let dy4D = dy as Tensor4D;
  let reshapedTo4D = false;
  if (dy.rank === 3) {
    reshapedTo4D = true;
    dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
  }
  const res = ENGINE.runKernel(
      backend => backend.depthwiseConv2DDerInput(dy4D, filter, convInfo),
      {dy4D});
  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }
  return res as T;
}

function depthwiseConv2dDerFilter<T extends Tensor3D|Tensor4D>(
    x: T, dy: T, filterShape: [number, number, number, number],
    convInfo: conv_util.Conv2DInfo): Tensor4D {
  let x4D = x as Tensor4D;
  if (x.rank === 3) {
    x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
  }
  let dy4D = dy as Tensor4D;
  if (dy4D.rank === 3) {
    dy4D = dy.as4D(1, dy.shape[0], dy.shape[1], dy.shape[2]);
  }
  return ENGINE.runKernel(
      backend => backend.depthwiseConv2DDerFilter(x4D, dy4D, convInfo),
      {x4D, dy4D});
}

/**
 * Computes a 3D convolution over the input x.
 *
 * @param x The input tensor, of rank 5 or rank 4, of shape
 *     `[batch, depth, height, width, channels]`. If rank 4,
 * batch of 1 is assumed.
 * @param filter The filter, rank 5, of shape
 *     `[filterDepth, filterHeight, filterWidth, inChannels, outChannels]`.
 *      inChannels must match between input and filter.
 * @param strides The strides of the convolution: `[strideDepth, strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dataFormat: An optional string from: "NDHWC", "NCDHW". Defaults to
 *     "NDHWC". Specify the data format of the input and output data. With the
 *     default format "NDHWC", the data is stored in the order of: [batch,
 *     depth, height, width, channels]. Only "NDHWC" is currently supported.
 * @param dilations The dilation rates: `[dilationDepth, dilationHeight,
 *     dilationWidth]` in which we sample input values across the height
 *     and width dimensions in atrous convolution. Defaults to `[1, 1, 1]`.
 *     If `dilations` is a single number, then
 *     `dilationDepth == dilationHeight == dilationWidth`. If it is greater
 *     than 1, then all values of `strides` must be 1.
 */

/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function conv3d_<T extends Tensor4D|Tensor5D>(
    x: T|TensorLike, filter: Tensor5D|TensorLike,
    strides: [number, number, number]|number, pad: 'valid'|'same',
    dataFormat: 'NDHWC'|'NCDHW' = 'NDHWC',
    dilations: [number, number, number]|number = [1, 1, 1]): T {
  const $x = convertToTensor(x, 'x', 'conv3d');
  const $filter = convertToTensor(filter, 'filter', 'conv3d');

  let x5D = $x as Tensor5D;
  let reshapedTo5D = false;

  if ($x.rank === 4) {
    reshapedTo5D = true;
    x5D = $x.as5D(1, $x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]);
  }
  util.assert(
      x5D.rank === 5,
      () => `Error in conv3d: input must be rank 5, but got rank ${x5D.rank}.`);
  util.assert(
      $filter.rank === 5,
      () => `Error in conv3d: filter must be rank 5, but got rank ` +
          `${$filter.rank}.`);
  util.assert(
      x5D.shape[4] === $filter.shape[3],
      () => `Error in conv3d: depth of input (${x5D.shape[4]}) must match ` +
          `input depth for filter ${$filter.shape[3]}.`);
  util.assert(
      eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in conv3D: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
  util.assert(
      dataFormat === 'NDHWC',
      () => `Error in conv3d: got dataFormat of ${
          dataFormat} but only NDHWC is currently supported.`);

  const convInfo = conv_util.computeConv3DInfo(
      x5D.shape, $filter.shape, strides, dilations, pad);

  const grad = (dy: Tensor5D, saved: Tensor[]) => {
    util.assert(
        tupleValuesAreOne(dilations),
        () =>
            'Error in gradient of conv3D: dilation rates greater than 1 are ' +
            `not yet supported in gradients. Got dilations '${dilations}'`);
    const [x5D, $filter] = saved;
    return {
      x: () => conv3dDerInput_(
          (x5D as Tensor5D).shape, dy, $filter as Tensor5D, strides, pad),
      $filter: () => conv3dDerFilter_(
          x5D as Tensor5D, dy, ($filter as Tensor5D).shape, strides, pad)
    };
  };

  const res = ENGINE.runKernel((backend, save) => {
    const res = backend.conv3d(x5D, $filter, convInfo);
    save([x5D, $filter]);
    return res;
  }, {x: x5D, $filter}, grad);
  if (reshapedTo5D) {
    return res.as4D(res.shape[1], res.shape[2], res.shape[3], res.shape[4]) as
        T;
  }
  return res as T;
}

/**
 * Computes the derivative of the input of a 3D convolution.
 *
 * @param xShape The shape of the input: [batch, depth, height, width,
 * in_channels]. If length of 4, batch of 1 is assumed.
 * @param dy The derivative of the output, of rank 5 or rank 4 of shape
 *   `[batch, outDepth, outHeight, outWidth, in_channels]`.
 * If rank 4, batch of 1 is assumed.
 * @param filter The filter, rank 5, of shape
 *     `[filterDepth, filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideDepth, strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm used:
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 */
function conv3dDerInput_<T extends Tensor4D|Tensor5D>(
    xShape:
        [number, number, number, number,
         number]|[number, number, number, number],
    dy: T, filter: Tensor5D, strides: [number, number, number]|number,
    pad: 'valid'|'same'): T {
  util.assert(
      xShape.length === dy.rank,
      () => `Length of inShape ` +
          `(${xShape.length}) and rank of dy (${dy.rank}) must match`);

  let xShape5D = xShape as [number, number, number, number, number];
  let dy5D = dy as Tensor5D;
  let reshapedTo5D = false;
  if (dy.rank === 4) {
    reshapedTo5D = true;
    dy5D = dy.as5D(1, dy.shape[0], dy.shape[1], dy.shape[2], dy.shape[3]);
    xShape5D = [1, xShape[0], xShape[1], xShape[2], xShape[3]];
  }

  const inDepth = xShape5D[4];
  const outDepth = dy5D.shape[4];
  util.assert(
      xShape5D.length === 5,
      () =>
          `Error in conv3dDerInput: inShape must be length 5, but got length ` +
          `${xShape5D.length}.`);
  util.assert(
      dy5D.rank === 5,
      () => `Error in conv3dDerInput: dy must be rank 5, but got ` +
          `rank ${dy5D.rank}`);
  util.assert(
      filter.rank === 5,
      () => `Error in conv3dDerInput: filter must be rank 5, but got ` +
          `rank ${filter.rank}`);
  util.assert(
      inDepth === filter.shape[3],
      () => `Error in conv3dDerInput: depth of input (${inDepth}) must ` +
          `match input depth for filter ${filter.shape[3]}.`);
  util.assert(
      outDepth === filter.shape[4],
      () => `Error in conv3dDerInput: depth of output (${outDepth}) must ` +
          `match output depth for filter ${filter.shape[4]}.`);

  const dilations = 1;

  const convInfo = conv_util.computeConv3DInfo(
      xShape5D, filter.shape, strides, dilations, pad);
  const res = ENGINE.runKernel(
      backend => backend.conv3dDerInput(dy5D, filter, convInfo), {dy5D});
  if (reshapedTo5D) {
    return res.as4D(res.shape[1], res.shape[2], res.shape[3], res.shape[4]) as
        T;
  }
  return res as T;
}

/**
 * Computes the derivative of the filter of a 3D convolution.
 *
 * @param x The input tensor, of rank 5 or rank 4 of shape
 *     [batch, depth, height, width, inChannels]. If rank 4, batch of 1 is
 *     assumed.
 * @param dy The dy image, of rank 5 or rank 4, of shape
 *     [batch, depth, height, width, outDepth]. If rank 4, batch of 1 is
 *     assumed.
 * @param filterShape The shape of the filter, length 5,
 *     [filterDepth, filterHeight, filterWidth, inDepth, outDepth].
 * @param strides The strides of the convolution: [strideDepth, strideHeight,
 * strideWidth].
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 */
function conv3dDerFilter_<T extends Tensor4D|Tensor5D>(
    x: T, dy: T, filterShape: [number, number, number, number, number],
    strides: [number, number, number]|number, pad: 'valid'|'same'): Tensor5D {
  let x5D = x as Tensor5D;
  if (x.rank === 4) {
    x5D = x.as5D(1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
  }
  let dy5D = dy as Tensor5D;
  if (dy5D.rank === 4) {
    dy5D = dy.as5D(1, dy.shape[0], dy.shape[1], dy.shape[2], dy.shape[3]);
  }
  util.assert(
      x5D.rank === 5,
      () => `Error in conv3dDerFilter: input must be rank 5, but got shape ` +
          `${x5D.shape}.`);
  util.assert(
      dy5D.rank === 5,
      () => `Error in conv3dDerFilter: dy must be rank 5, but got shape ` +
          `${dy5D.shape}.`);
  util.assert(
      filterShape.length === 5,
      () => `Error in conv3dDerFilter: filterShape must be length 5, but got ` +
          `${filterShape}.`);
  util.assert(
      x5D.shape[4] === filterShape[3],
      () => `Error in conv3dDerFilter: depth of input ${x5D.shape[4]}) must ` +
          `match input depth in filter (${filterShape[3]}.`);
  util.assert(
      dy5D.shape[4] === filterShape[4],
      () => `Error in conv3dDerFilter: depth of dy (${dy5D.shape[4]}) must ` +
          `match output depth for filter (${filterShape[4]}).`);

  const dilations = 1;

  const convInfo = conv_util.computeConv3DInfo(
      x5D.shape, filterShape, strides, dilations, pad);
  return ENGINE.runKernel(
      backend => backend.conv3dDerFilter(x5D, dy5D, convInfo), {x5D, dy5D});
}

export const conv1d = op({conv1d_});
export const conv2d = op({conv2d_});
export const conv3d = op({conv3d_});
export const conv2dDerFilter = op({conv2dDerFilter_});
export const conv2dDerInput = op({conv2dDerInput_});
export const depthwiseConv2d = op({depthwiseConv2d_});
export const separableConv2d = op({separableConv2d_});
export const conv2dTranspose = op({conv2dTranspose_});
