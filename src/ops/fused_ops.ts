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
import {conv2dDerFilter, conv2dDerInput} from '../ops/conv';
import * as conv_util from '../ops/conv_util';
import {op} from '../ops/operation';
import {Tensor, Tensor3D, Tensor4D} from '../tensor';
import {makeTypesMatch} from '../tensor_util';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import * as broadcast_util from './broadcast_util';
import {Activation} from './fused_util';

/**
 * Computes the dot product of two matrices with optional activation and bias.
 *
 * ```js
 * const a = tf.tensor2d([-1, -2], [1, 2]);
 * const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 * const bias = tf.tensor2d([1, 2], [1, 2]);
 *
 * tf.fused.matMul(a, b, false, false, bias, 'relu').print();
 * ```
 *
 * @param a First matrix in dot product operation.
 * @param b Second matrix in dot product operation.
 * @param transposeA If true, `a` is transposed before multiplication.
 * @param transposeB If true, `b` is transposed before multiplication.
 * @param bias Matrix to be added to the result.
 * @param activation Name of activation kernel (defaults to `linear`).
 */
/** @doc {heading: 'Operations', subheading: 'Matrices', namespace: 'fused'} */
function matMul_<T extends Tensor>(
    a: T|TensorLike, b: T|TensorLike, transposeA = false, transposeB = false,
    bias?: Tensor|TensorLike, activation: Activation = 'linear'): T {
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

  const grad = (dy: Tensor3D, saved: Tensor[]) => {
    const [a3D, b3D, y] = saved;

    let dyActivation: Tensor3D;
    if (activation == null || activation === 'linear') {
      dyActivation = dy;
    } else if (activation === 'relu') {
      dyActivation = dy.mul(y.step()) as Tensor3D;
    } else {
      throw new Error(
          `Gradient for activation ${activation} has not been ` +
          `implemented yet.`);
    }

    let biasGradient = {};
    if (bias != null) {
      biasGradient = {
        $bias: () => {
          let res = dyActivation;
          // Using dyActivation as reference shape because outputShape does not
          // account for the fact that we temporarily reshape inputs to 3D as
          // part of batched matMul.
          const reduceAxes =
              broadcast_util.getReductionAxes($bias.shape, dyActivation.shape);
          if (reduceAxes.length > 0) {
            res = res.sum(reduceAxes);
          }
          return res.reshape($bias.shape);
        }
      };
    }

    if (!transposeA && !transposeB) {
      return Object.assign(
          {
            $a: () => dyActivation.matMul(b3D as Tensor3D, false, true),
            $b: () => a3D.matMul(dyActivation, true, false)
          },
          biasGradient);
    } else if (!transposeA && transposeB) {
      return Object.assign(
          {
            $a: () => dyActivation.matMul(b3D as Tensor3D, false, false),
            $b: () => dyActivation.matMul(a3D as Tensor3D, true, false)
          },
          biasGradient);
    } else if (transposeA && !transposeB) {
      return Object.assign(
          {
            $a: () => b3D.matMul(dyActivation, false, true),
            $b: () => a3D.matMul(dyActivation, false, false)
          },
          biasGradient);
    } else {
      return Object.assign(
          {
            $a: () => b3D.matMul(dyActivation, true, true),
            $b: () => dyActivation.matMul(a3D as Tensor3D, true, true)
          },
          biasGradient);
    }
  };

  const inputs: {$a: Tensor, $b: Tensor, $bias?: Tensor} = {$a: a3D, $b: b3D};
  if (bias != null) {
    inputs.$bias = $bias;
  }

  const res = ENGINE.runKernel((backend, save) => {
    const y = backend.fusedBatchMatMul(
        a3D, b3D, transposeA, transposeB, $bias, activation);
    save([a3D, b3D, y]);
    return y;
  }, inputs, grad);
  return res.reshape(outShape) as T;
}

/**
 * Computes a 2D convolution over the input x, optionally fused with adding a
 * bias and applying an activation.
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
 * @param bias Tensor to be added to the result.
 * @param activation Name of activation kernel (defaults to `linear`).
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function conv2d_<T extends Tensor3D|Tensor4D>(
    x: T|TensorLike, filter: Tensor4D|TensorLike,
    strides: [number, number]|number, pad: 'valid'|'same'|number,
    dataFormat: 'NHWC'|'NCHW' = 'NHWC',
    dilations: [number, number]|number = [1, 1],
    dimRoundingMode?: 'floor'|'round'|'ceil', bias?: Tensor|TensorLike,
    activation: Activation = 'linear'): T {
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

  const grad = (dy: Tensor4D, saved: Tensor[]) => {
    const [$filter, x4D, y] = saved as [Tensor4D, Tensor4D, Tensor4D];

    let dyActivation: Tensor4D;
    if (activation == null || activation === 'linear') {
      dyActivation = dy;
    } else if (activation === 'relu') {
      dyActivation = dy.mul(y.step()) as Tensor4D;
    } else {
      throw new Error(
          `Gradient for activation ${activation} has not been ` +
          `implemented yet.`);
    }

    util.assert(
        conv_util.tupleValuesAreOne(dilations),
        () => 'Error in gradient of fused conv2D: ' +
            `dilation rates greater than 1 ` +
            `are not yet supported in gradients. Got dilations '${dilations}'`);

    let biasGradient = {};
    if (bias != null) {
      biasGradient = {
        $bias: () => {
          let res = dyActivation;
          const reduceAxes =
              broadcast_util.getReductionAxes($bias.shape, dyActivation.shape);
          if (reduceAxes.length > 0) {
            res = res.sum(reduceAxes);
          }
          return res.reshape($bias.shape);
        }
      };
    }

    return Object.assign(
        {
          x: () =>
              conv2dDerInput(x4D.shape, dyActivation, $filter, strides, pad),
          $filter: () =>
              conv2dDerFilter(x4D, dyActivation, $filter.shape, strides, pad)
        },
        biasGradient);
  };

  const inputs: {x: Tensor, $filter: Tensor,
                 $bias?: Tensor} = {x: x4D, $filter};
  if (bias != null) {
    inputs.$bias = $bias;
  }

  const res = ENGINE.runKernel((backend, save) => {
    const res = backend.fusedConv2d(
        x4D, $filter, convInfo, $bias as Tensor4D, activation);
    save([$filter, x4D, res]);

    return res;
  }, inputs, grad);

  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }
  return res as T;
}

export const matMul = op({matMul_});
export const conv2d = op({conv2d_});

export {Activation};
