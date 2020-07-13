/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {ENGINE} from '../../engine';
import * as conv_util from '../../ops/conv_util';
import {op} from '../../ops/operation';
import {Tensor, Tensor3D, Tensor4D} from '../../tensor';
import {makeTypesMatch} from '../../tensor_util';
import {convertToTensor} from '../../tensor_util_env';
import {TensorLike} from '../../types';
import * as util from '../../util';
import {add} from '../add';
import * as broadcast_util from '../broadcast_util';
import {depthwiseConv2d as unfusedDepthwiseConv2d} from '../depthwise_conv2d';
import {depthwiseConv2dNativeBackpropFilter} from '../depthwise_conv2d_native_backprop_filter';
import {depthwiseConv2dNativeBackpropInput} from '../depthwise_conv2d_native_backprop_input';
import {Activation, applyActivation, getFusedBiasGradient, getFusedDyActivation, shouldFuse} from '../fused_util';

/**
 * Computes depthwise 2D convolution, optionally fused with adding a
 * bias and applying an activation.
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
 * @param obj An object with the following properties:
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
 * @param bias Tensor to be added to the result.
 * @param activation Name of activation kernel (defaults to `linear`).
 * @param preluActivationWeights Tensor of prelu weights to be applied as part
 *     of a `prelu` activation, typically the same shape as `x`.
 */
function fusedDepthwiseConv2d_<T extends Tensor3D|Tensor4D>({
  x,
  filter,
  strides,
  pad,
  dataFormat = 'NHWC',
  dilations = [1, 1],
  dimRoundingMode,
  bias,
  activation = 'linear',
  preluActivationWeights
}: {
  x: T|TensorLike,
  filter: Tensor4D|TensorLike,
  strides: [number, number]|number,
  pad: 'valid'|'same'|number,
  dataFormat?: 'NHWC'|'NCHW',
  dilations?: [number, number]|number,
  dimRoundingMode?: 'floor'|'round'|'ceil',
  bias?: Tensor|TensorLike,
  activation?: Activation,
  preluActivationWeights?: Tensor
}): T {
  if (shouldFuse(ENGINE.state.gradientDepth, activation) === false) {
    let result = unfusedDepthwiseConv2d(
        x, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
    if (bias != null) {
      result = add(result, bias);
    }

    return applyActivation(result, activation, preluActivationWeights) as T;
  }

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
      () => `Error in fused depthwiseConv2d: input must be rank 4, but got ` +
          `rank ${x4D.rank}.`);
  util.assert(
      $filter.rank === 4,
      () => `Error in fused depthwiseConv2d: filter must be rank 4, ` +
          `but got rank ${$filter.rank}.`);
  util.assert(
      x4D.shape[3] === $filter.shape[2],
      () => `Error in fused depthwiseConv2d: number of input channels ` +
          `(${x4D.shape[3]}) must match the inChannels dimension in ` +
          `filter ${$filter.shape[2]}.`);
  if (dilations == null) {
    dilations = [1, 1];
  }
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () =>
          'Error in fused depthwiseConv2d: Either strides or dilations must ' +
          `be 1. Got strides ${strides} and dilations '${dilations}'`);

  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in fused depthwiseConv2d: pad must be an integer when ` +
            `using dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const convInfo = conv_util.computeConv2DInfo(
      x4D.shape, $filter.shape, strides, dilations, pad, dimRoundingMode,
      true /* depthwise */);

  let $bias: Tensor;
  if (bias != null) {
    $bias = convertToTensor(bias, 'bias', 'fused conv2d');
    [$bias] = makeTypesMatch($bias, $x);

    broadcast_util.assertAndGetBroadcastShape(convInfo.outShape, $bias.shape);
  }

  let $preluActivationWeights: Tensor;
  if (preluActivationWeights != null) {
    $preluActivationWeights = convertToTensor(
        preluActivationWeights, 'prelu weights', 'fused depthwiseConv2d');
  }

  const grad = (dy: Tensor4D, saved: Tensor[]) => {
    util.assert(
        conv_util.tupleValuesAreOne(dilations),
        () => 'Error in gradient of fused depthwiseConv2d: dilation rates ' +
            `greater than 1 are not yet supported. Got dilations ` +
            `'${dilations}'`);
    const [$filter, x4D, y] = saved;

    const dyActivation = getFusedDyActivation(dy, y, activation) as Tensor4D;

    let biasGradient = {};
    if (bias != null) {
      biasGradient = {bias: () => getFusedBiasGradient($bias, dyActivation)};
    }

    return Object.assign(
        {
          x: () => depthwiseConv2dNativeBackpropInput(
              (x4D as Tensor4D).shape, dyActivation, $filter as Tensor4D,
              convInfo),
          filter: () => depthwiseConv2dNativeBackpropFilter(
              x4D as Tensor4D, dyActivation, ($filter as Tensor4D).shape,
              convInfo),
        },
        biasGradient);
  };

  const inputs: {
    x: Tensor,
    filter: Tensor,
    bias?: Tensor,
    preluActivationWeights?: Tensor
  } = {x: x4D, filter: $filter};
  if (bias != null) {
    inputs.bias = $bias;
  }
  if (preluActivationWeights != null) {
    inputs.preluActivationWeights = $preluActivationWeights;
  }

  const inputsToSave = [$filter, x4D];
  const outputsToSave = [true];
  const res = ENGINE.runKernelFunc(
      (backend, save) => {
        const res = backend.fusedDepthwiseConv2D({
          input: x4D,
          filter: $filter,
          convInfo,
          bias: $bias,
          activation,
          preluActivationWeights: $preluActivationWeights
        });
        save([$filter, x4D, res]);
        return res;
      },
      inputs, grad, 'FusedDepthwiseConv2D', {convInfo, activation},
      inputsToSave, outputsToSave);
  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }
  return res as T;
}
export const depthwiseConv2d = op({fusedDepthwiseConv2d_});
