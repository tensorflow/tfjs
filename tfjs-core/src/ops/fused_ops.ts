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

import {ENGINE} from '../engine';
import {conv2dDerFilter, conv2dDerInput, depthwiseConv2dDerFilter, depthwiseConv2dDerInput} from '../ops/conv';
import * as conv_util from '../ops/conv_util';
import {op} from '../ops/operation';
import {Tensor, Tensor3D, Tensor4D} from '../tensor';
import {makeTypesMatch} from '../tensor_util';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {add} from './binary_ops';
import * as broadcast_util from './broadcast_util';
import {conv2d as unfusedConv2d, depthwiseConv2d as unfusedDepthwiseConv2d} from './conv';
import {Activation, shouldFuse} from './fused_util';
import {matMul as unfusedMatMul} from './matmul';

import {elu, prelu, relu, relu6} from './relu_ops';

// Returns gradient for fused activation.
const getFusedDyActivation =
    (dy: Tensor, y: Tensor, activation: Activation): Tensor => {
      if (activation == null || activation === 'linear') {
        return dy;
      }
      if (activation === 'relu') {
        return dy.mul(y.step());
      }
      throw new Error(
          `Gradient for activation ${activation} has not been ` +
          `implemented yet.`);
    };

// Returns gradient for fused bias.
const getFusedBiasGradient = (bias: Tensor, dyActivation: Tensor): Tensor => {
  let res = dyActivation;
  const reduceAxes =
      broadcast_util.getReductionAxes(bias.shape, dyActivation.shape);
  if (reduceAxes.length > 0) {
    res = res.sum(reduceAxes);
  }
  return res.reshape(bias.shape);
};

const applyActivation =
    (x: Tensor, activation: Activation, preluActivationWeights?: Tensor):
        Tensor => {
          if (activation === 'linear') {
            return x;
          } else if (activation === 'relu') {
            return relu(x);
          } else if (activation === 'elu') {
            return elu(x);
          } else if (activation === 'relu6') {
            return relu6(x);
          } else if (activation === 'prelu') {
            return prelu(x, preluActivationWeights);
          }
          throw new Error(`Unknown fused activation ${activation}.`);
        };

/**
 * Computes the dot product of two matrices with optional activation and bias.
 *
 * ```js
 * const a = tf.tensor2d([-1, -2], [1, 2]);
 * const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const bias = tf.tensor2d([1, 2], [1, 2]);
 *
 * tf.fused.matMul({a, b, bias, activation: 'relu'}).print();
 * ```
 *
 * @param obj An object with the following properties:
 * - `a` First matrix in dot product operation.
 * - `b` Second matrix in dot product operation.
 * - `transposeA` If true, `a` is transposed before multiplication.
 * - `transposeB` If true, `b` is transposed before multiplication.
 * - `bias` Matrix to be added to the result.
 * - `activation` Name of activation kernel (defaults to `linear`).
 * - `preluActivationWeights` Tensor of prelu weights.
 */
function fusedMatMul_<T extends Tensor>({
  a,
  b,
  transposeA = false,
  transposeB = false,
  bias,
  activation = 'linear',
  preluActivationWeights
}: {
  a: T|TensorLike,
  b: T|TensorLike,
  transposeA?: boolean,
  transposeB?: boolean,
  bias?: Tensor|TensorLike,
  activation?: Activation,
  preluActivationWeights?: Tensor
}): T {
  if (shouldFuse(ENGINE.state.gradientDepth, activation) === false) {
    let result = unfusedMatMul(a, b, transposeA, transposeB);
    if (bias != null) {
      result = add(result, bias);
    }

    return applyActivation(result, activation, preluActivationWeights) as T;
  }

  let $a = convertToTensor(a, 'a', 'fused matMul');
  let $b = convertToTensor(b, 'b', 'fused matMul');
  [$a, $b] = makeTypesMatch($a, $b);

  const innerShapeA =
      transposeA ? $a.shape[$a.rank - 2] : $a.shape[$a.rank - 1];
  const innerShapeB =
      transposeB ? $b.shape[$b.rank - 1] : $b.shape[$b.rank - 2];

  const outerShapeA =
      transposeA ? $a.shape[$a.rank - 1] : $a.shape[$a.rank - 2];
  const outerShapeB =
      transposeB ? $b.shape[$b.rank - 2] : $b.shape[$b.rank - 1];

  const outerDimsA = $a.shape.slice(0, -2);
  const outerDimsB = $b.shape.slice(0, -2);
  const batchDimA = util.sizeFromShape(outerDimsA);
  const batchDimB = util.sizeFromShape(outerDimsB);

  util.assert(
      $a.rank >= 2 && $b.rank >= 2 && $a.rank === $b.rank,
      () =>
          `Error in fused matMul: inputs must have the same rank of at least ` +
          `2, got ranks ${$a.rank} and ${$b.rank}.`);

  util.assert(
      util.arraysEqual(outerDimsA, outerDimsB),
      () => `Error in fused matMul: outer dimensions (${outerDimsA}) and (` +
          `${outerDimsB}) of Tensors with shapes ${$a.shape} and ` +
          `${$b.shape} must match.`);

  util.assert(
      innerShapeA === innerShapeB,
      () => `Error in fused matMul: inner shapes (${innerShapeA}) and (` +
          `${innerShapeB}) of Tensors with shapes ${$a.shape} and ` +
          `${$b.shape} and transposeA=${transposeA}` +
          ` and transposeB=${transposeB} must match.`);

  const outShape = $a.shape.slice(0, -2).concat([outerShapeA, outerShapeB]);

  const a3D = transposeA ? $a.as3D(batchDimA, innerShapeA, outerShapeA) :
                           $a.as3D(batchDimA, outerShapeA, innerShapeA);
  const b3D = transposeB ? $b.as3D(batchDimB, outerShapeB, innerShapeB) :
                           $b.as3D(batchDimB, innerShapeB, outerShapeB);

  let $bias: Tensor;
  if (bias != null) {
    $bias = convertToTensor(bias, 'bias', 'fused matMul');
    [$bias] = makeTypesMatch($bias, $a);

    broadcast_util.assertAndGetBroadcastShape(outShape, $bias.shape);
  }

  let $preluActivationWeights: Tensor;
  if (preluActivationWeights != null) {
    $preluActivationWeights = convertToTensor(
        preluActivationWeights, 'prelu weights', 'fused matMul');
  }

  const grad = (dy: Tensor3D, saved: Tensor[]) => {
    const [a3D, b3D, y] = saved;
    const dyActivation = getFusedDyActivation(dy, y, activation);

    let biasGradient = {};
    if (bias != null) {
      biasGradient = {bias: () => getFusedBiasGradient($bias, dyActivation)};
    }

    if (!transposeA && !transposeB) {
      return Object.assign(
          {
            a: () => dyActivation.matMul(b3D as Tensor3D, false, true),
            b: () => a3D.matMul(dyActivation, true, false)
          },
          biasGradient);
    } else if (!transposeA && transposeB) {
      return Object.assign(
          {
            a: () => dyActivation.matMul(b3D as Tensor3D, false, false),
            b: () => dyActivation.matMul(a3D as Tensor3D, true, false)
          },
          biasGradient);
    } else if (transposeA && !transposeB) {
      return Object.assign(
          {
            a: () => b3D.matMul(dyActivation, false, true),
            b: () => a3D.matMul(dyActivation, false, false)
          },
          biasGradient);
    } else {
      return Object.assign(
          {
            a: () => b3D.matMul(dyActivation, true, true),
            b: () => dyActivation.matMul(a3D as Tensor3D, true, true)
          },
          biasGradient);
    }
  };

  const inputs:
      {a: Tensor, b: Tensor,
       bias?: Tensor,
       preluActivationWeights?: Tensor} = {a: a3D, b: b3D};
  if (bias != null) {
    inputs.bias = $bias;
  }
  if (preluActivationWeights != null) {
    inputs.preluActivationWeights = $preluActivationWeights;
  }

  const inputsToSave = [a3D, b3D];
  const outputsToSave = [true];

  const res = ENGINE.runKernelFunc(
      (backend, save) => {
        const y = backend.fusedBatchMatMul({
          a: a3D,
          b: b3D,
          transposeA,
          transposeB,
          bias: $bias,
          activation,
          preluActivationWeights: $preluActivationWeights
        });
        save([a3D, b3D, y]);
        return y;
      },
      inputs, grad, '_FusedMatMul', {transposeA, transposeB, activation},
      inputsToSave, outputsToSave);
  return res.reshape(outShape) as T;
}

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
  pad: 'valid'|'same'|number,
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
    x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
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
    const [$filter, x4D, y] = saved as [Tensor4D, Tensor4D, Tensor4D];

    const dyActivation = getFusedDyActivation(dy, y, activation) as Tensor4D;

    util.assert(
        conv_util.tupleValuesAreOne(dilations),
        () => 'Error in gradient of fused conv2D: ' +
            `dilation rates greater than 1 ` +
            `are not yet supported in gradients. Got dilations '${dilations}'`);

    let biasGradient = {};
    if (bias != null) {
      biasGradient = {bias: () => getFusedBiasGradient($bias, dyActivation)};
    }

    return Object.assign(
        {
          x: () =>
              conv2dDerInput(x4D.shape, dyActivation, $filter, strides, pad),
          filter: () =>
              conv2dDerFilter(x4D, dyActivation, $filter.shape, strides, pad)
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
  const outputsToSave = [true];  // Save the only output.
  const res = ENGINE.runKernelFunc(
      (backend, save) => {
        const res = backend.fusedConv2d({
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
      inputs, grad, 'FusedConv2D', {convInfo, activation}, inputsToSave,
      outputsToSave);

  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }

  return res as T;
}

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
          x: () => depthwiseConv2dDerInput(
              (x4D as Tensor4D).shape, dyActivation, $filter as Tensor4D,
              convInfo),
          filter: () => depthwiseConv2dDerFilter(
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

export const matMul = op({fusedMatMul_});
export const conv2d = op({fusedConv2d_});
export const depthwiseConv2d = op({fusedDepthwiseConv2d_});

export {Activation};
