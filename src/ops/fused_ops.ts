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
import {op} from '../ops/operation';
import {Tensor, Tensor3D} from '../tensor';
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
 * const c = tf.tensor2d([1, 2], [1, 2]);
 *
 * tf.fused.matMul(a, b, false, false, 'relu', c);
 * ```
 *
 * @param a First matrix in dot product operation.
 * @param b Second matrix in dot product operation.
 * @param transposeA If true, `a` is transposed before multiplication.
 * @param transposeB If true, `b` is transposed before multiplication.
 * @param activation Name of activation kernel (defaults to `linear`).
 * @param bias Matrix to be added to the result.
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

export const matMul = op({matMul_});

export {Activation};
