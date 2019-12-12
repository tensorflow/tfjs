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
import {Tensor, Tensor1D, Tensor2D, Tensor3D} from '../tensor';
import {makeTypesMatch} from '../tensor_util';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';
import {op} from './operation';

/**
 * Computes the dot product of two matrices, A * B. These must be matrices.
 *
 * ```js
 * const a = tf.tensor2d([1, 2], [1, 2]);
 * const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * a.matMul(b).print();  // or tf.matMul(a, b)
 * ```
 * @param a First matrix in dot product operation.
 * @param b Second matrix in dot product operation.
 * @param transposeA If true, `a` is transposed before multiplication.
 * @param transposeB If true, `b` is transposed before multiplication.
 */
/** @doc {heading: 'Operations', subheading: 'Matrices'} */
function matMul_<T extends Tensor>(
    a: T|TensorLike, b: T|TensorLike, transposeA = false,
    transposeB = false): T {
  let $a = convertToTensor(a, 'a', 'matMul');
  let $b = convertToTensor(b, 'b', 'matMul');
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
      () => `Error in matMul: inputs must have the same rank of at least 2, ` +
          `got ranks ${$a.rank} and ${$b.rank}.`);

  util.assert(
      util.arraysEqual(outerDimsA, outerDimsB),
      () => `Error in matMul: outer dimensions (${outerDimsA}) and (` +
          `${outerDimsB}) of Tensors with shapes ${$a.shape} and ` +
          `${$b.shape} must match.`);

  util.assert(
      innerShapeA === innerShapeB,
      () => `Error in matMul: inner shapes (${innerShapeA}) and (` +
          `${innerShapeB}) of Tensors with shapes ${$a.shape} and ` +
          `${$b.shape} and transposeA=${transposeA}` +
          ` and transposeB=${transposeB} must match.`);

  const outShape = $a.shape.slice(0, -2).concat([outerShapeA, outerShapeB]);

  const a3D = transposeA ? $a.as3D(batchDimA, innerShapeA, outerShapeA) :
                           $a.as3D(batchDimA, outerShapeA, innerShapeA);
  const b3D = transposeB ? $b.as3D(batchDimB, outerShapeB, innerShapeB) :
                           $b.as3D(batchDimB, innerShapeB, outerShapeB);

  const grad = (dy: Tensor3D, saved: Tensor[]) => {
    const [a3D, b3D] = saved as Tensor3D[];
    if (!transposeA && !transposeB) {
      return {
        a: () => dy.matMul(b3D, false, true),
        b: () => a3D.matMul(dy, true, false)
      };
    } else if (!transposeA && transposeB) {
      return {
        a: () => dy.matMul(b3D, false, false),
        b: () => dy.matMul(a3D, true, false)
      };
    } else if (transposeA && !transposeB) {
      return {
        a: () => b3D.matMul(dy, false, true),
        b: () => a3D.matMul(dy, false, false)
      };
    } else {
      return {
        a: () => b3D.matMul(dy, true, true),
        b: () => dy.matMul(a3D, true, true)
      };
    }
  };

  const attrs = {transposeA, transposeB};
  const res = ENGINE.runKernelFunc((backend, save) => {
    const res = backend.batchMatMul(a3D, b3D, transposeA, transposeB);
    save([a3D, b3D]);
    return res;
  }, {a: a3D, b: b3D}, grad, 'BatchMatMul', attrs);
  return res.reshape(outShape) as T;
}

/**
 * Computes the outer product of two vectors, `v1` and `v2`.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 * const b = tf.tensor1d([3, 4, 5]);
 *
 * tf.outerProduct(a, b).print();
 * ```
 * @param v1 The first vector in the outer product operation.
 * @param v2 The second vector in the outer product operation.
 */
/** @doc {heading: 'Operations', subheading: 'Matrices'} */
function outerProduct_(
    v1: Tensor1D|TensorLike, v2: Tensor1D|TensorLike): Tensor2D {
  const $v1 = convertToTensor(v1, 'v1', 'outerProduct');
  const $v2 = convertToTensor(v2, 'v2', 'outerProduct');

  util.assert(
      $v1.rank === 1 && $v2.rank === 1,
      () => `Error in outerProduct: inputs must be rank 1, but got ranks ` +
          `${$v1.rank} and ${$v2.rank}.`);

  return $v1.as2D(-1, 1).matMul($v2.as2D(1, -1));
}

/**
 * Computes the dot product of two matrices and/or vectors, `t1` and `t2`.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor2d([[1, 2], [3, 4]]);
 * const c = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
 *
 * a.dot(b).print();  // or tf.dot(a, b)
 * b.dot(a).print();
 * b.dot(c).print();
 * ```
 * @param t1 The first tensor in the dot operation.
 * @param t2 The second tensor in the dot operation.
 */
/** @doc {heading: 'Operations', subheading: 'Matrices'} */
function dot_(t1: Tensor|TensorLike, t2: Tensor|TensorLike): Tensor {
  const $t1 = convertToTensor(t1, 't1', 'dot');
  const $t2 = convertToTensor(t2, 't2', 'dot');
  util.assert(
      ($t1.rank === 1 || $t1.rank === 2) && ($t2.rank === 1 || $t2.rank === 2),
      () => `Error in dot: inputs must all be rank 1 or 2, but got ranks ` +
          `${$t1.rank} and ${$t2.rank}.`);

  const t1Inner = ($t1.rank === 1 ? $t1.size : $t1.shape[1]);
  const t2Inner = ($t2.rank === 1 ? $t2.size : $t2.shape[0]);

  util.assert(
      t1Inner === t2Inner,
      () => `Error in dot: inner dimensions of inputs must match, but got ` +
          `${t1Inner} and ${t2Inner}.`);

  if ($t1.rank === 1 && $t2.rank === 1) {
    return $t1.as2D(1, -1).matMul($t2.as2D(-1, 1)).asScalar();
  } else if ($t1.rank === 1 && $t2.rank === 2) {
    return $t1.as2D(1, -1).matMul($t2.as2D($t2.shape[0], $t2.shape[1])).as1D();
  } else if ($t1.rank === 2 && $t2.rank === 1) {
    return $t1.matMul($t2.as2D(-1, 1)).as1D();
  } else {
    return $t1.matMul($t2.as2D($t2.shape[0], $t2.shape[1]));
  }
}

export const matMul = op({matMul_});
export const dot = op({dot_});
export const outerProduct = op({outerProduct_});
