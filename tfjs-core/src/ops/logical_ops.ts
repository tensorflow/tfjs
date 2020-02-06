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

import {whereImpl} from '../backends/where_impl';
import {ENGINE} from '../engine';
import {Tensor, Tensor2D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {assert, assertShapesMatch} from '../util';
import {assertAndGetBroadcastShape} from './broadcast_util';
import {op} from './operation';
import {zerosLike} from './tensor_ops';

/**
 * Returns the truth value of `NOT x` element-wise.
 *
 * ```js
 * const a = tf.tensor1d([false, true], 'bool');
 *
 * a.logicalNot().print();
 * ```
 *
 * @param x The input tensor. Must be of dtype 'bool'.
 */
/** @doc {heading: 'Operations', subheading: 'Logical'} */
function logicalNot_<T extends Tensor>(x: T|TensorLike): T {
  const $x = convertToTensor(x, 'x', 'logicalNot', 'bool');
  return ENGINE.runKernelFunc(backend => backend.logicalNot($x), {$x});
}

/**
 * Returns the truth value of `a AND b` element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([false, false, true, true], 'bool');
 * const b = tf.tensor1d([false, true, false, true], 'bool');
 *
 * a.logicalAnd(b).print();
 * ```
 *
 * @param a The first input tensor. Must be of dtype bool.
 * @param b The second input tensor. Must be of dtype bool.
 */
/** @doc {heading: 'Operations', subheading: 'Logical'} */
function logicalAnd_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  const $a = convertToTensor(a, 'a', 'logicalAnd', 'bool');
  const $b = convertToTensor(b, 'b', 'logicalAnd', 'bool');
  assertAndGetBroadcastShape($a.shape, $b.shape);

  return ENGINE.runKernelFunc(
             backend => backend.logicalAnd($a, $b), {a: $a, b: $b},
             null /* grad */, 'LogicalAnd') as T;
}

/**
 * Returns the truth value of `a OR b` element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([false, false, true, true], 'bool');
 * const b = tf.tensor1d([false, true, false, true], 'bool');
 *
 * a.logicalOr(b).print();
 * ```
 * @param a The first input tensor. Must be of dtype bool.
 * @param b The second input tensor. Must be of dtype bool.
 */
/** @doc {heading: 'Operations', subheading: 'Logical'} */
function logicalOr_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  const $a = convertToTensor(a, 'a', 'logicalOr', 'bool');
  const $b = convertToTensor(b, 'b', 'logicalOr', 'bool');
  assertAndGetBroadcastShape($a.shape, $b.shape);

  return ENGINE.runKernelFunc(backend => backend.logicalOr($a, $b), {$a, $b}) as
      T;
}

/**
 * Returns the truth value of `a XOR b` element-wise. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([false, false, true, true], 'bool');
 * const b = tf.tensor1d([false, true, false, true], 'bool');
 *
 * a.logicalXor(b).print();
 * ```
 *
 * @param a The first input tensor. Must be of dtype bool.
 * @param b The second input tensor. Must be of dtype bool.
 */
/** @doc {heading: 'Operations', subheading: 'Logical'} */
function logicalXor_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  const $a = convertToTensor(a, 'a', 'logicalXor', 'bool');
  const $b = convertToTensor(b, 'b', 'logicalXor', 'bool');
  assertAndGetBroadcastShape($a.shape, $b.shape);

  // x ^ y = (x | y) & ~(x & y)
  return logicalOr(a, b).logicalAnd(logicalAnd(a, b).logicalNot()) as T;
}

/**
 * Returns the elements, either `a` or `b` depending on the `condition`.
 *
 * If the condition is true, select from `a`, otherwise select from `b`.
 *
 * ```js
 * const cond = tf.tensor1d([false, false, true], 'bool');
 * const a = tf.tensor1d([1 , 2, 3]);
 * const b = tf.tensor1d([-1, -2, -3]);
 *
 * a.where(cond, b).print();
 * ```
 *
 * @param condition The input condition. Must be of dtype bool.
 * @param a If `condition` is rank 1, `a` may have a higher rank but
 *     its first dimension must match the size of `condition`.
 * @param b A tensor with the same shape and type as `a`.
 */
/** @doc {heading: 'Operations', subheading: 'Logical'} */
function where_<T extends Tensor>(
    condition: Tensor|TensorLike, a: T|TensorLike, b: T|TensorLike): T {
  const $a = convertToTensor(a, 'a', 'where');
  const $b = convertToTensor(b, 'b', 'where');
  const $condition = convertToTensor(condition, 'condition', 'where', 'bool');

  assertShapesMatch($a.shape, $b.shape, 'Error in where: ');

  if ($condition.rank === 1) {
    // If condition rank is 1, then the first dimension must match the size of
    // condition.
    assert(
        $condition.shape[0] === $a.shape[0],
        () => 'The first dimension of `a` must match the size of `condition`.');
  } else {
    // A must have the same shape as condition.
    assertShapesMatch($condition.shape, $b.shape, 'Error in where: ');
  }

  // TODO(julianoks): Return null for condition gradient
  // when backprop supports it.
  const grad = (dy: T, saved: Tensor[]) => {
    const [$condition] = saved;
    return {
      $condition: () => zerosLike($condition).toFloat(),
      $a: () => dy.mul($condition.cast(dy.dtype)),
      $b: () => dy.mul($condition.logicalNot().cast(dy.dtype))
    } as {$a: () => T, $b: () => T, $condition: () => T};
  };

  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.select($condition, $a, $b);
    save([$condition]);
    return res;
  }, {$condition, $a, $b}, grad) as T;
}

/**
 * Returns the coordinates of true elements of condition.
 *
 * The coordinates are returned in a 2-D tensor where the first dimension (rows)
 * represents the number of true elements, and the second dimension (columns)
 * represents the coordinates of the true elements. Keep in mind, the shape of
 * the output tensor can vary depending on how many true values there are in
 * input. Indices are output in row-major order. The resulting tensor has the
 * shape `[numTrueElems, condition.rank]`.
 *
 * This is analogous to calling the python `tf.where(cond)` without an x or y.
 *
 * ```js
 * const cond = tf.tensor1d([false, false, true], 'bool');
 * const result = await tf.whereAsync(cond);
 * result.print();
 * ```
 */
/** @doc {heading: 'Operations', subheading: 'Logical'} */
async function whereAsync_(condition: Tensor|TensorLike): Promise<Tensor2D> {
  const $condition =
      convertToTensor(condition, 'condition', 'whereAsync', 'bool');
  const vals = await $condition.data();
  const res = whereImpl($condition.shape, vals);
  if (condition !== $condition) {
    $condition.dispose();
  }
  return res;
}

export const logicalAnd = op({logicalAnd_});
export const logicalNot = op({logicalNot_});
export const logicalOr = op({logicalOr_});
export const logicalXor = op({logicalXor_});
export const where = op({where_});
export const whereAsync = whereAsync_;
