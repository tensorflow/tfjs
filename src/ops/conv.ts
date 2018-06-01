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

import {doc} from '../doc';
import {ENV} from '../environment';
import {Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import * as util from '../util';
import * as conv_util from './conv_util';
import {operation} from './operation';

export class ConvOps {
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
  @doc({heading: 'Operations', subheading: 'Convolution'})
  @operation
  static conv1d<T extends Tensor2D|Tensor3D>(
      x: T, filter: Tensor3D, stride: number, pad: 'valid'|'same'|number,
      dataFormat: 'NWC'|'NCW' = 'NWC', dilation = 1,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    util.assertArgumentsAreTensors({x, filter}, 'conv1d');

    let x3D = x as Tensor3D;
    let reshapedTo3D = false;
    if (x.rank === 2) {
      reshapedTo3D = true;
      x3D = x.as3D(1, x.shape[0], x.shape[1]);
    }

    util.assert(
        x3D.rank === 3,
        `Error in conv1d: input must be rank 3, but got rank ${x3D.rank}.`);
    util.assert(
        filter.rank === 3,
        `Error in conv1d: filter must be rank 3, but got rank ` +
            `${filter.rank}.`);
    if (dimRoundingMode != null) {
      util.assert(
          util.isInt(pad as number),
          `Error in conv1d: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
    }

    util.assert(
        x3D.shape[2] === filter.shape[1],
        `Error in conv1d: depth of input (${x3D.shape[2]}) must match ` +
            `input depth for filter ${filter.shape[1]}.`);
    util.assert(
        eitherStridesOrDilationsAreOne(stride, dilation),
        'Error in conv1D: Either stride or dilation must be 1. ' +
            `Got stride ${stride} and dilation '${dilation}'`);
    util.assert(
        dataFormat === 'NWC',
        `Error in conv1d: got dataFormat of ${
            dataFormat} but only NWC is currently supported.`);

    const filter4D =
        filter.as4D(1, filter.shape[0], filter.shape[1], filter.shape[2]);
    const input4D = x3D.as4D(x3D.shape[0], 1, x3D.shape[1], x3D.shape[2]);
    const strides: [number, number] = [1, stride];
    const dilations: [number, number] = [1, dilation];

    const conv2dDataFormat = 'NHWC';

    const res = ConvOps.conv2d(
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
  @doc({heading: 'Operations', subheading: 'Convolution'})
  @operation
  static conv2d<T extends Tensor3D|Tensor4D>(
      x: T, filter: Tensor4D, strides: [number, number]|number,
      pad: 'valid'|'same'|number, dataFormat: 'NHWC'|'NCHW' = 'NHWC',
      dilations: [number, number]|number = [1, 1],
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    util.assertArgumentsAreTensors({x, filter}, 'conv2d');

    let x4D = x as Tensor4D;
    let reshapedTo4D = false;

    if (x.rank === 3) {
      reshapedTo4D = true;
      x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
    }
    util.assert(
        x4D.rank === 4,
        `Error in conv2d: input must be rank 4, but got rank ${x4D.rank}.`);
    util.assert(
        filter.rank === 4,
        `Error in conv2d: filter must be rank 4, but got rank ` +
            `${filter.rank}.`);
    if (dimRoundingMode != null) {
      util.assert(
          util.isInt(pad as number),
          `Error in conv2d: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
    }

    util.assert(
        x4D.shape[3] === filter.shape[2],
        `Error in conv2d: depth of input (${x4D.shape[3]}) must match ` +
            `input depth for filter ${filter.shape[2]}.`);
    util.assert(
        eitherStridesOrDilationsAreOne(strides, dilations),
        'Error in conv2D: Either strides or dilations must be 1. ' +
            `Got strides ${strides} and dilations '${dilations}'`);
    util.assert(
        dataFormat === 'NHWC',
        `Error in conv2d: got dataFormat of ${
            dataFormat} but only NHWC is currently supported.`);

    const convInfo = conv_util.computeConv2DInfo(
        x4D.shape, filter.shape, strides, dilations, pad, dimRoundingMode);

    const grad = (dy: Tensor4D) => {
      util.assert(
          tupleValuesAreOne(dilations),
          'Error in gradient of conv2D: dilation rates greater than 1 are not' +
              `yet supported in gradients. Got dilations '${dilations}'`);

      return {
        x: () => ConvOps.conv2dDerInput(x4D.shape, dy, filter, strides, pad),
        filter: () =>
            ConvOps.conv2dDerFilter(x4D, dy, filter.shape, strides, pad)
      };
    };

    const res = ENV.engine.runKernel(
        backend => backend.conv2d(x4D, filter, convInfo), {x: x4D, filter},
        grad);
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
  @operation
  static conv2dDerInput<T extends Tensor3D|Tensor4D>(
      xShape: [number, number, number, number]|[number, number, number], dy: T,
      filter: Tensor4D, strides: [number, number]|number,
      pad: 'valid'|'same'|number, dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    util.assertArgumentsAreTensors({dy, filter}, 'conv2dDerInput');

    util.assert(
        xShape.length === dy.rank,
        `Length of inShape ` +
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
        `Error in conv2dDerInput: inShape must be length 4, but got length ` +
            `${xShape4D.length}.`);
    util.assert(
        dy4D.rank === 4,
        `Error in conv2dDerInput: dy must be rank 4, but got ` +
            `rank ${dy4D.rank}`);
    util.assert(
        filter.rank === 4,
        `Error in conv2dDerInput: filter must be rank 4, but got ` +
            `rank ${filter.rank}`);
    util.assert(
        inDepth === filter.shape[2],
        `Error in conv2dDerInput: depth of input (${inDepth}) must ` +
            `match input depth for filter ${filter.shape[2]}.`);
    util.assert(
        outDepth === filter.shape[3],
        `Error in conv2dDerInput: depth of output (${outDepth}) must ` +
            `match output depth for filter ${filter.shape[3]}.`);
    if (dimRoundingMode != null) {
      util.assert(
          util.isInt(pad as number),
          `Error in conv2dDerInput: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
    }

    const dilations = 1;

    const convInfo = conv_util.computeConv2DInfo(
        xShape4D, filter.shape, strides, dilations, pad, dimRoundingMode);
    const res = ENV.engine.runKernel(
        backend => backend.conv2dDerInput(dy4D, filter, convInfo), {dy4D});
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
  @operation
  static conv2dDerFilter<T extends Tensor3D|Tensor4D>(
      x: T, dy: T, filterShape: [number, number, number, number],
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): Tensor4D {
    util.assertArgumentsAreTensors({x, dy}, 'conv2dDerFilter');

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
        `Error in conv2dDerFilter: input must be rank 4, but got shape ` +
            `${x4D.shape}.`);
    util.assert(
        dy4D.rank === 4,
        `Error in conv2dDerFilter: dy must be rank 4, but got shape ` +
            `${dy4D.shape}.`);
    util.assert(
        filterShape.length === 4,
        `Error in conv2dDerFilter: filterShape must be length 4, but got ` +
            `${filterShape}.`);
    util.assert(
        x4D.shape[3] === filterShape[2],
        `Error in conv2dDerFilter: depth of input ${x4D.shape[3]}) must ` +
            `match input depth in filter (${filterShape[2]}.`);
    util.assert(
        dy4D.shape[3] === filterShape[3],
        `Error in conv2dDerFilter: depth of dy (${dy4D.shape[3]}) must ` +
            `match output depth for filter (${filterShape[3]}).`);
    if (dimRoundingMode != null) {
      util.assert(
          util.isInt(pad as number),
          `Error in conv2dDerFilter: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
    }

    const dilations = 1;

    const convInfo = conv_util.computeConv2DInfo(
        x4D.shape, filterShape, strides, dilations, pad, dimRoundingMode);
    return ENV.engine.runKernel(
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
  @doc({heading: 'Operations', subheading: 'Convolution'})
  @operation
  static conv2dTranspose<T extends Tensor3D|Tensor4D>(
      x: T, filter: Tensor4D,
      outputShape: [number, number, number, number]|[number, number, number],
      strides: [number, number]|number, pad: 'valid'|'same'|number,
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    util.assertArgumentsAreTensors({x, filter}, 'conv2dTranspose');

    return ConvOps.conv2dDerInput(
        outputShape, x, filter, strides, pad, dimRoundingMode);
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
  @doc({heading: 'Operations', subheading: 'Convolution'})
  @operation
  static depthwiseConv2d<T extends Tensor3D|Tensor4D>(
      x: T, filter: Tensor4D, strides: [number, number]|number,
      pad: 'valid'|'same'|number, dataFormat: 'NHWC'|'NCHW' = 'NHWC',
      dilations: [number, number]|number = [1, 1],
      dimRoundingMode?: 'floor'|'round'|'ceil'): T {
    util.assertArgumentsAreTensors({x, filter}, 'depthwiseConv2d');

    let x4D = x as Tensor4D;
    let reshapedTo4D = false;
    if (x.rank === 3) {
      reshapedTo4D = true;
      x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
    }
    util.assert(
        x4D.rank === 4,
        `Error in depthwiseConv2d: input must be rank 4, but got ` +
            `rank ${x4D.rank}.`);
    util.assert(
        filter.rank === 4,
        `Error in depthwiseConv2d: filter must be rank 4, but got rank ` +
            `${filter.rank}.`);
    util.assert(
        x4D.shape[3] === filter.shape[2],
        `Error in depthwiseConv2d: number of input channels ` +
            `(${x4D.shape[3]}) must match the inChannels dimension in ` +
            `filter ${filter.shape[2]}.`);
    if (dilations == null) {
      dilations = [1, 1];
    }
    util.assert(
        eitherStridesOrDilationsAreOne(strides, dilations),
        'Error in depthwiseConv2d: Either strides or dilations must be 1. ' +
            `Got strides ${strides} and dilations '${dilations}'`);

    if (dimRoundingMode != null) {
      util.assert(
          util.isInt(pad as number),
          `Error in depthwiseConv2d: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
    }

    const convInfo = conv_util.computeConv2DInfo(
        x4D.shape, filter.shape, strides, dilations, pad, dimRoundingMode,
        true /* depthwise */);

    const grad = (dy: Tensor4D) => {
      util.assert(
          tupleValuesAreOne(dilations),
          'Error in gradient of depthwiseConv2d: dilation rates greater than ' +
              `1 are not yet supported. Got dilations '${dilations}'`);
      return {
        x: () => depthwiseConv2dDerInput(x4D.shape, dy, filter, convInfo),
        filter: () => depthwiseConv2dDerFilter(x4D, dy, filter.shape, convInfo),
      };
    };

    const res = ENV.engine.runKernel(
        backend => backend.depthwiseConv2D(x4D, filter, convInfo),
        {x: x4D, filter}, grad);
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
  @doc({heading: 'Operations', subheading: 'Convolution'})
  @operation
  static separableConv2d<T extends Tensor3D|Tensor4D>(
      x: T, depthwiseFilter: Tensor4D, pointwiseFilter: Tensor4D,
      strides: [number, number]|number, pad: 'valid'|'same',
      dilation: [number, number]|number = [1, 1],
      dataFormat: 'NHWC'|'NCHW' = 'NHWC'): T {
    util.assertArgumentsAreTensors(
        {x, depthwiseFilter, pointwiseFilter}, 'separableConv2d');

    let x4D = x as Tensor4D;
    let reshapedTo4D = false;
    if (x.rank === 3) {
      reshapedTo4D = true;
      x4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
    }

    if (dataFormat === 'NCHW') {
      throw new Error(
          'separableConv2d currently does not support dataFormat NCHW; only ' +
          'NHWC is supported');
    }

    util.assert(
        x4D.rank === 4,
        `Error in separableConv2d: input must be rank 4, but got ` +
            `rank ${x4D.rank}.`);
    util.assert(
        depthwiseFilter.rank === 4,
        `Error in separableConv2d: depthwise filter must be rank 4, but got ` +
            `rank ${depthwiseFilter.rank}.`);
    util.assert(
        pointwiseFilter.rank === 4,
        `Error in separableConv2d: pointwise filter must be rank 4, but got ` +
            `rank ${depthwiseFilter.rank}.`);
    util.assert(
        pointwiseFilter.shape[0] === 1,
        `Error in separableConv2d: the first dimension of pointwise filter ` +
            ` must be 1, but got ${pointwiseFilter.shape[0]}.`);
    util.assert(
        pointwiseFilter.shape[1] === 1,
        `Error in separableConv2d: the second dimension of pointwise filter ` +
            ` must be 1, but got ${pointwiseFilter.shape[1]}.`);

    const inChannels = depthwiseFilter.shape[2];
    const channelMultiplier = depthwiseFilter.shape[3];
    util.assert(
        pointwiseFilter.shape[2] === inChannels * channelMultiplier,
        `Error in separableConv2d: the third dimension of pointwise filter ` +
            `must be ${inChannels * channelMultiplier}, ` +
            `but got ${pointwiseFilter.shape[2]}.`);

    const depthwise = ConvOps.depthwiseConv2d(
        x4D, depthwiseFilter, strides, pad, dataFormat, dilation);
    const pointwiseStride = 1;
    const res = ConvOps.conv2d(
        depthwise, pointwiseFilter, pointwiseStride, 'valid', dataFormat);
    if (reshapedTo4D) {
      return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
    }
    return res as T;
  }
}

function parseTupleParam(param: number|[number, number]): [number, number] {
  return typeof param === 'number' ? [param, param] : param;
}

function tupleValuesAreOne(param: number|[number, number]): boolean {
  const [dimA, dimB] = parseTupleParam(param);
  return dimA === 1 && dimB === 1;
}

function eitherStridesOrDilationsAreOne(
    strides: number|[number, number],
    dilations: number|[number, number]): boolean {
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
  const res = ENV.engine.runKernel(
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
  return ENV.engine.runKernel(
      backend => backend.depthwiseConv2DDerFilter(x4D, dy4D, convInfo),
      {x4D, dy4D});
}
