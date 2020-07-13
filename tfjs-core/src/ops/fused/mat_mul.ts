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
import {op} from '../../ops/operation';
import {Tensor, Tensor3D} from '../../tensor';
import {makeTypesMatch} from '../../tensor_util';
import {convertToTensor} from '../../tensor_util_env';
import {TensorLike} from '../../types';
import * as util from '../../util';
import {add} from '../add';
import * as broadcast_util from '../broadcast_util';
import {applyActivation, getFusedBiasGradient, getFusedDyActivation, shouldFuse} from '../fused_util';
import {matMul as unfusedMatMul} from '../mat_mul';
import {Activation} from './types';

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
  return res.reshape(outShape);
}

export const matMul = op({fusedMatMul_});
