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
import {_FusedMatMul, _FusedMatMulAttrs, _FusedMatMulInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor3D} from '../tensor';
import {GradSaveFunc, NamedTensorMap} from '../tensor_types';
import {makeTypesMatch} from '../tensor_util';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {add} from './add';
import * as broadcast_util from './broadcast_util';
import {Activation} from './fused_types';
import {applyActivation, getFusedBiasGradient, getFusedDyActivation, shouldFuse} from './fused_util';
import {matMul as unfusedMatMul} from './mat_mul';
import {op} from './operation';
import {reshape} from './reshape';

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

  const a3D: Tensor3D = transposeA ?
      reshape($a, [batchDimA, innerShapeA, outerShapeA]) :
      reshape($a, [batchDimA, outerShapeA, innerShapeA]);
  const b3D: Tensor3D = transposeB ?
      reshape($b, [batchDimB, outerShapeB, innerShapeB]) :
      reshape($b, [batchDimB, innerShapeB, outerShapeB]);

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
    const [a3D, b3D, y, $bias] = saved;
    // we reshape dy because the result of the forward is not
    // necessarily going to be a 3d tensor due to a reshape done at the end of
    // the customOp.
    const dyActivation =
        getFusedDyActivation(reshape(dy, y.shape), y, activation);
    let aDer: Tensor;
    let bDer: Tensor;

    if (!transposeA && !transposeB) {
      aDer = unfusedMatMul(dyActivation, b3D, false, true);
      bDer = unfusedMatMul(a3D, dyActivation, true, false);
    } else if (!transposeA && transposeB) {
      aDer = unfusedMatMul(dyActivation, b3D, false, false);
      bDer = unfusedMatMul(dyActivation, a3D, true, false);
    } else if (transposeA && !transposeB) {
      aDer = unfusedMatMul(b3D, dyActivation, false, true);
      bDer = unfusedMatMul(a3D, dyActivation, false, false);
    } else {
      aDer = unfusedMatMul(b3D, dyActivation, true, true);
      bDer = unfusedMatMul(dyActivation, a3D, true, true);
    }

    if (bias != null) {
      const biasDer = getFusedBiasGradient($bias, dyActivation);
      return [aDer, bDer, biasDer];
    } else {
      return [aDer, bDer];
    }
  };

  const forward: ForwardFunc<Tensor> = (backend) => {
    const y = backend.fusedBatchMatMul({
      a: a3D,
      b: b3D,
      transposeA,
      transposeB,
      bias: $bias,
      activation,
      preluActivationWeights: $preluActivationWeights
    });
    return y;
  };

  const inputs: _FusedMatMulInputs = {
    a: a3D,
    b: b3D,
    bias: $bias,
    preluActivationWeights: $preluActivationWeights
  };
  const attrs: _FusedMatMulAttrs = {transposeA, transposeB, activation};

  // Depending on the the params passed in we will have different number of
  // inputs and thus a a different number of elements in the gradient.
  if (bias == null) {
    const customOp =
        customGrad((a3D: Tensor3D, b3D: Tensor3D, save: GradSaveFunc) => {
          const res = ENGINE.runKernelFunc(
              forward, inputs as {} as NamedTensorMap, null /* grad */,
              _FusedMatMul, attrs as {} as NamedAttrMap);

          save([a3D, b3D, res]);

          return {value: reshape(res, outShape), gradFunc: grad};
        });
    return customOp(a3D, b3D) as T;
  } else {
    const customOpWithBias = customGrad(
        (a3D: Tensor3D, b3D: Tensor3D, $bias: Tensor, save: GradSaveFunc) => {
          const res = ENGINE.runKernelFunc(
              forward, inputs as {} as NamedTensorMap, null /* grad */,
              _FusedMatMul, attrs as {} as NamedAttrMap);

          save([a3D, b3D, res, $bias]);

          return {value: reshape(res, outShape), gradFunc: grad};
        });

    return customOpWithBias(a3D, b3D, $bias) as T;
  }
}

export const matMul = op({fusedMatMul_});
