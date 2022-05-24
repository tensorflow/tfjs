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
import {customGrad} from '../../gradients';
import {FusedConv2D, FusedConv2DAttrs, FusedConv2DInputs} from '../../kernel_names';
import {NamedAttrMap} from '../../kernel_registry';
import {Tensor, Tensor3D, Tensor4D} from '../../tensor';
import {GradSaveFunc, NamedTensorMap} from '../../tensor_types';
import {makeTypesMatch} from '../../tensor_util';
import {convertToTensor} from '../../tensor_util_env';
import {TensorLike} from '../../types';
import * as util from '../../util';
import {add} from '../add';
import * as broadcast_util from '../broadcast_util';
import {conv2d as unfusedConv2d} from '../conv2d';
import {conv2DBackpropFilter} from '../conv2d_backprop_filter';
import {conv2DBackpropInput} from '../conv2d_backprop_input';
import * as conv_util from '../conv_util';
import {Activation} from '../fused_types';
import {applyActivation, getFusedBiasGradient, getFusedDyActivation, shouldFuse} from '../fused_util';
import {op} from '../operation';
import {reshape} from '../reshape';

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
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dataFormat An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels]. Only "NHWC" is currently supported.
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in atrous convolution. Defaults to `[1, 1]`. If `dilations` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @param bias Tensor to be added to the result.
 * @param activation Name of activation kernel (defaults to `linear`) to be
 *     applied
 *      after biasAdd.
 * @param preluActivationWeights Tensor of prelu weights to be applied as part
 *     of a `prelu` activation, typically the same shape as `x`.
 * @param leakyreluAlpha Optional. Alpha to be applied as part of a `leakyrelu`
 *     activation.
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
  preluActivationWeights,
  leakyreluAlpha
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
  preluActivationWeights?: Tensor,
  leakyreluAlpha?: number
}): T {
  activation = activation || 'linear';

  if (shouldFuse(ENGINE.state.gradientDepth, activation) === false) {
    // TODO: Transpose bias and preluActivationWeights properly for NCHW
    // format before computation.
    util.assert(
        dataFormat === 'NHWC',
        () => `Error in fused conv2d: got dataFormat of ${dataFormat} but ` +
            `only NHWC is currently supported for the case of gradient depth ` +
            `is 0 and the activation is not linear.`);

    let result = unfusedConv2d(
        x, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
    if (bias != null) {
      result = add(result, bias);
    }

    return applyActivation(
               result, activation, preluActivationWeights, leakyreluAlpha) as T;
  }

  const $x = convertToTensor(x, 'x', 'conv2d', 'float32');
  const $filter = convertToTensor(filter, 'filter', 'conv2d', 'float32');

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
  conv_util.checkPadOnDimRoundingMode('fused conv2d', pad, dimRoundingMode);
  const inputChannels = dataFormat === 'NHWC' ? x4D.shape[3] : x4D.shape[1];
  util.assert(
      $filter.shape[2] === inputChannels,
      () => `Error in conv2d: depth of input (${inputChannels}) must match ` +
          `input depth for filter ${$filter.shape[2]}.`);
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in conv2D: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);

  const convInfo = conv_util.computeConv2DInfo(
      x4D.shape, $filter.shape, strides, dilations, pad, dimRoundingMode);

  let $bias: Tensor;
  if (bias != null) {
    $bias = convertToTensor(bias, 'bias', 'fused conv2d');
    [$bias] = makeTypesMatch($bias, $x);

    // According to TensorFlow, the bias is supposed be a 1-D tensor or a
    // scalar.
    //
    // 3-D or 4-D bias is not disabled for NHWC format, because they are
    // currently being used in some cases. For examplem in our code base,
    // https://github.com/tensorflow/tfjs/blob/b53bd47e880367ae57493f0ea628abaf08db2d5d/tfjs-core/src/ops/fused/fused_conv2d_test.ts#L1972.
    if (dataFormat === 'NHWC') {
      broadcast_util.assertAndGetBroadcastShape(convInfo.outShape, $bias.shape);
    } else {
      util.assert(
          $bias.shape.length <= 1,
          () => `Error in fused conv2d: only supports scalar or 1-D Tensor ` +
              `bias for NCHW format but got the bias of ` +
              `rank-${$bias.shape.length}.`);

      util.assert(
          $bias.shape.length === 0 || $bias.shape[0] === convInfo.outChannels ||
              $bias.shape[0] === 1,
          () => `Error in fused conv2d: bias shape (${$bias.shape}) is not ` +
              `compatible with the number of output channels ` +
              `(${convInfo.outChannels})`);
    }
  }

  let $preluActivationWeights: Tensor;
  if (preluActivationWeights != null) {
    // PReLU's activation weights could be a scalar, a 1-D tensor or a 3-D
    // tensor.
    const alphaShape = preluActivationWeights.shape;
    util.assert(
        alphaShape.length <= 1 || alphaShape.length === 3,
        () => `Error in fused conv2d: only supports scalar, 1-D Tensor or ` +
            `3-D Tensor PReLU activation weights but got a tensor of ` +
            `rank-${alphaShape.length}.`);

    if (alphaShape.length === 1) {
      // Whether the data format is NCHW or NHWC, the 1-D PReLU activation
      // weights tensor should be aligned with the output channels of conv2d
      // result.
      util.assert(
          alphaShape[0] === 1 || alphaShape[0] === convInfo.outChannels,
          () => `Error in fused conv2d: PReLU activation weights ` +
              `(${alphaShape}) is not compatible with the number of output ` +
              `channels (${convInfo.outChannels}).`);
    } else if (alphaShape.length === 3) {
      // Whether the data format is NCHW or NHWC, the PReLU activation weights
      // tensor should has the compatible shape with the result of conv2d.
      try {
        broadcast_util.assertAndGetBroadcastShape(
            alphaShape, convInfo.outShape);
      } catch (e) {
        const errMsg =
            `Error in fused conv2d: PReLU activation weights (${alphaShape}) ` +
            `is not compatible with the output shape of the conv2d ` +
            `(${convInfo.outShape}).`;
        throw Error(errMsg);
      }
    }

    $preluActivationWeights = convertToTensor(
        preluActivationWeights, 'prelu weights', 'fused conv2d');
  }

  const grad = (dy: Tensor4D, saved: Tensor[]) => {
    util.assert(
        dataFormat === 'NHWC',
        () => `Error in gradient of fused conv2D: got dataFormat of ${
            dataFormat} but only NHWC is currently supported.`);

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

  const inputs: FusedConv2DInputs = {
    x: x4D,
    filter: $filter,
    bias: $bias,
    preluActivationWeights: $preluActivationWeights
  };

  const attrs: FusedConv2DAttrs = {
    strides,
    pad,
    dataFormat,
    dilations,
    dimRoundingMode,
    activation,
    leakyreluAlpha
  };

  // Depending on the the params passed in we will have different number of
  // inputs and thus a a different number of elements in the gradient.
  if (bias == null) {
    const customOp =
        customGrad((x4D: Tensor4D, filter: Tensor4D, save: GradSaveFunc) => {
          let res: Tensor4D|Tensor3D =
              // tslint:disable-next-line: no-unnecessary-type-assertion
              ENGINE.runKernel(
                  FusedConv2D, inputs as {} as NamedTensorMap,
                  attrs as {} as NamedAttrMap);

          save([filter, x4D, res]);

          if (reshapedTo4D) {
            // tslint:disable-next-line: no-unnecessary-type-assertion
            res = reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as
                Tensor3D;
          }

          return {value: res, gradFunc: grad};
        });
    return customOp(x4D, $filter) as T;
  } else {
    const customOpWithBias = customGrad(
        (x4D: Tensor4D, filter: Tensor4D, bias: Tensor, save: GradSaveFunc) => {
          let res: Tensor4D|Tensor3D = ENGINE.runKernel(
              FusedConv2D, inputs as {} as NamedTensorMap,
              attrs as {} as NamedAttrMap);

          save([filter, x4D, res, bias]);

          if (reshapedTo4D) {
            // tslint:disable-next-line: no-unnecessary-type-assertion
            res = reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as
                Tensor3D;
          }

          return {value: res, gradFunc: grad};
        });

    return customOpWithBias(x4D, $filter, $bias) as T;
  }
}
export const conv2d = op({fusedConv2d_});
