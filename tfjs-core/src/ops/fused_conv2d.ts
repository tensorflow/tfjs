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

import {ENGINE, ForwardFunc} from '../engine';
import {customGrad} from '../gradients';
import {FusedConv2D, FusedConv2DAttrs, FusedConv2DInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {conv2DBackpropFilter} from '../ops/conv2d_backprop_filter';
import {conv2DBackpropInput} from '../ops/conv2d_backprop_input';
import {Tensor, Tensor3D, Tensor4D} from '../tensor';
import {GradSaveFunc, NamedTensorMap} from '../tensor_types';
import {makeTypesMatch} from '../tensor_util';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {add} from './add';
import * as broadcast_util from './broadcast_util';
import {conv2d as unfusedConv2d} from './conv2d';
import * as conv_util from './conv_util';
import {Activation} from './fused_types';
import {applyActivation, getFusedBiasGradient, getFusedDyActivation, shouldFuse} from './fused_util';
import {op} from './operation';
import {reshape} from './reshape';

/**
 * Computes a 2D convolution over the input x, optionally fused with adding a
 * bias and applying an activation.
 *
 * ```js
 * const inputDepth = 2;
 * const inShape = [2, 2, 2, inputDepth];
 * const outputDepth = 2;
 * const fSize = 1;
 * const pad = 0;
 * const strides = 1;
 *
 * const x = tf.tensor4d( [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
 * 16], inShape);
 * const w = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth,
 * outputDepth]);
 *
 * tf.fused.conv2d({ x, filter: w, strides, pad, dataFormat: 'NHWC',
 * dilations: [1, 1], bias: tf.scalar(5), activation: 'relu' }).print();
 * ```
 *
 * @param obj An object with the following properties:
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter, rank 4, of shape
 *     `[filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm.
 *   - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *   - `valid` output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dataFormat An optional string from: "NHWC", "NCHW". Defaults to
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
 * @param bias Tensor to be added to the result.
 * @param activation Name of activation kernel (defaults to `linear`) to be
 *     applied
 *      after biasAdd.
 * @param preluActivationWeights Tensor of prelu weights to be applied as part
 *     of a `prelu` activation, typically the same shape as `x`.
 */
function fusedConv2d_<T extends Tensor3D|Tensor4D>({
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
  pad: 'valid'|'same'|number|conv_util.ExplicitPadding,
  dataFormat?: 'NHWC'|'NCHW',
  dilations?: [number, number]|number,
  dimRoundingMode?: 'floor'|'round'|'ceil',
  bias?: Tensor|TensorLike,
  activation?: Activation,
  preluActivationWeights?: Tensor
}): T {
  activation = activation || 'linear';

  if (shouldFuse(ENGINE.state.gradientDepth, activation) === false) {
    let result = unfusedConv2d(
        x, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
    if (bias != null) {
      result = add(result, bias);
    }

    return applyActivation(result, activation, preluActivationWeights) as T;
  }

  const $x = convertToTensor(x, 'x', 'conv2d');
  const $filter = convertToTensor(filter, 'filter', 'conv2d');

  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;

  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }
  util.assert(
      x4D.rank === 4,
      () => `Error in fused conv2d: input must be rank 4, but got rank ` +
          `${x4D.rank}.`);
  util.assert(
      $filter.rank === 4,
      () => `Error in fused conv2d: filter must be rank 4, but got rank ` +
          `${$filter.rank}.`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in fused conv2d: pad must be an integer when using, ` +
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

  let $bias: Tensor;
  if (bias != null) {
    $bias = convertToTensor(bias, 'bias', 'fused conv2d');
    [$bias] = makeTypesMatch($bias, $x);

    broadcast_util.assertAndGetBroadcastShape(convInfo.outShape, $bias.shape);
  }

  let $preluActivationWeights: Tensor;
  if (preluActivationWeights != null) {
    $preluActivationWeights = convertToTensor(
        preluActivationWeights, 'prelu weights', 'fused conv2d');
  }

  const grad = (dy: Tensor4D, saved: Tensor[]) => {
    const [$filter, x4D, y, $bias] =
        saved as [Tensor4D, Tensor4D, Tensor4D, Tensor];

    const dyActivation = getFusedDyActivation(dy, y, activation) as Tensor4D;

    util.assert(
        conv_util.tupleValuesAreOne(dilations),
        () => 'Error in gradient of fused conv2D: ' +
            `dilation rates greater than 1 ` +
            `are not yet supported in gradients. Got dilations '${dilations}'`);

    const xDer =
        conv2DBackpropInput(x4D.shape, dyActivation, $filter, strides, pad);
    const filterDer =
        conv2DBackpropFilter(x4D, dyActivation, $filter.shape, strides, pad);
    const der: Tensor[] = [xDer, filterDer];

    if ($bias != null) {
      const biasDer = getFusedBiasGradient($bias, dyActivation);
      der.push(biasDer);
    }
    return der;
  };

  const forward: ForwardFunc<Tensor> = (backend) => {
    const res = backend.fusedConv2d({
      input: x4D,
      filter: $filter,
      convInfo,
      bias: $bias,
      activation,
      preluActivationWeights: $preluActivationWeights
    });
    return res;
  };

  const inputs: FusedConv2DInputs = {
    x: x4D,
    filter: $filter,
    bias: $bias,
    preluActivationWeights: $preluActivationWeights
  };

  const attrs: FusedConv2DAttrs =
      {strides, pad, dataFormat, dilations, dimRoundingMode, activation};

  // Depending on the the params passed in we will have different number of
  // inputs and thus a a different number of elements in the gradient.
  if (bias == null) {
    const customOp =
        customGrad((x4D: Tensor4D, filter: Tensor4D, save: GradSaveFunc) => {
          let res = ENGINE.runKernelFunc(
              forward, inputs as {} as NamedTensorMap, null /* grad */,
              FusedConv2D, attrs as {} as NamedAttrMap);

          save([filter, x4D, res]);

          if (reshapedTo4D) {
            res = reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as T;
          }

          return {value: res, gradFunc: grad};
        });
    return customOp(x4D, $filter) as T;
  } else {
    const customOpWithBias = customGrad(
        (x4D: Tensor4D, filter: Tensor4D, bias: Tensor, save: GradSaveFunc) => {
          let res = ENGINE.runKernelFunc(
              forward, inputs as {} as NamedTensorMap, null /* grad */,
              FusedConv2D, attrs as {} as NamedAttrMap);

          save([filter, x4D, res, bias]);

          if (reshapedTo4D) {
            res = reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as T;
          }

          return {value: res, gradFunc: grad};
        });

    return customOpWithBias(x4D, $filter, $bias) as T;
  }
}
export const conv2d = op({fusedConv2d_});
