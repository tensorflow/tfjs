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
import {deprecationWarn} from '../globals';
import {Tensor} from '../tensor';
import {makeTypesMatch} from '../tensor_util';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import * as broadcast_util from './broadcast_util';
import {op} from './operation';

/**
 * @deprecated
 * Adds two `tf.Tensor`s element-wise, A + B.
 *
 * Inputs must be the same shape. For broadcasting support, use add() instead.
 *
 * @param a The first Tensor to add element-wise.
 * @param b The second Tensor to add element-wise.
 */
function addStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');
  const $a = convertToTensor(a, 'a', 'addStrict');
  const $b = convertToTensor(b, 'b', 'addStrict');
  util.assertShapesMatch($a.shape, $b.shape, 'Error in addStrict: ');
  return $a.add($b);
}

/**
 * @deprecated
 * Subtracts two `tf.Tensor`s element-wise, A - B. Inputs must
 * be the same shape.
 *
 * For broadcasting support, use `tf.sub` instead.
 *
 * @param a The first Tensor to subtract element-wise.
 * @param b The second Tensor to subtract element-wise.
 */
function subStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');

  const $a = convertToTensor(a, 'a', 'subStrict');
  const $b = convertToTensor(b, 'b', 'subStrict');
  util.assertShapesMatch($a.shape, $b.shape, 'Error in subStrict: ');
  return $a.sub($b);
}

/**
 * @deprecated
 * Computes the power of one `tf.Tensor` to another. Inputs must
 * be the same shape.
 *
 * For broadcasting support, use `tf.pow` instead.
 *
 * @param base The base tensor to pow element-wise.
 * @param exp The exponent tensor to pow element-wise.
 */
function powStrict_<T extends Tensor>(base: T, exp: Tensor): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');

  util.assertShapesMatch(base.shape, exp.shape, 'Error in powStrict: ');
  return base.pow(exp);
}

/**
 * Multiplies two `tf.Tensor`s element-wise, A * B. Supports broadcasting.
 *
 * We also expose `tf.mulStrict` which has the same signature as this op and
 * asserts that `a` and `b` are the same shape (does not broadcast).
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3, 4]);
 * const b = tf.tensor1d([2, 3, 4, 5]);
 *
 * a.mul(b).print();  // or tf.mul(a, b)
 * ```
 *
 * ```js
 * // Broadcast mul a with b.
 * const a = tf.tensor1d([1, 2, 3, 4]);
 * const b = tf.scalar(5);
 *
 * a.mul(b).print();  // or tf.mul(a, b)
 * ```
 * @param a The first tensor to multiply.
 * @param b The second tensor to multiply. Must have the same dtype as `a`.
 */
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function mul_<T extends Tensor>(a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'mul');
  let $b = convertToTensor(b, 'b', 'mul');
  [$a, $b] = makeTypesMatch($a, $b);

  const outShape =
      broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);

  const der = (dy: Tensor, saved: Tensor[]) => {
    const [$a, $b] = saved;
    const derA = () => {
      const res = dy.mul($b.toFloat());
      const reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
      if (reduceAxes.length > 0) {
        return res.sum(reduceAxes).reshape($a.shape);
      }
      return res;
    };
    const derB = () => {
      const res = dy.mul($a.toFloat());
      const reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
      if (reduceAxes.length > 0) {
        return res.sum(reduceAxes).reshape($b.shape);
      }
      return res;
    };
    return {a: derA, b: derB};
  };
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.multiply($a, $b);
    save([$a, $b]);
    return res;
  }, {a: $a, b: $b}, der, 'Mul') as T;
}

/**
 * @deprecated
 * Multiplies two `tf.Tensor`s element-wise, A * B.
 *
 * Inputs must be the same shape. For broadcasting support, use `tf.mul`.
 *
 * @param a The first tensor to multiply.
 * @param b The first tensor to multiply. Must have the same
 *    dtype as `a`.
 */
function mulStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');

  const $a = convertToTensor(a, 'a', 'mul');
  const $b = convertToTensor(b, 'b', 'mul');
  util.assertShapesMatch($a.shape, $b.shape, 'Error in multiplyStrict: ');
  return $a.mul($b);
}

/**
 * @deprecated
 * Divides two `tf.Tensor`s element-wise, A / B. Inputs must
 * be the same shape.
 *
 * @param a The first tensor as the numerator for element-wise division.
 * @param b The second tensor as the denominator for element-wise division.
 */
function divStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');

  const $a = convertToTensor(a, 'a', 'div');
  const $b = convertToTensor(b, 'b', 'div');
  util.assertShapesMatch($a.shape, $b.shape, 'Error in divideStrict: ');
  return $a.div($b);
}

/**
 * Returns the mod of a and b element-wise.
 * `floor(x / y) * y + mod(x, y) = x`
 * Supports broadcasting.
 *
 * We also expose `tf.modStrict` which has the same signature as this op and
 * asserts that `a` and `b` are the same shape (does not broadcast).
 *
 * ```js
 * const a = tf.tensor1d([1, 4, 3, 16]);
 * const b = tf.tensor1d([1, 2, 9, 4]);
 *
 * a.mod(b).print();  // or tf.mod(a, b)
 * ```
 *
 * ```js
 * // Broadcast a mod b.
 * const a = tf.tensor1d([2, 4, 6, 8]);
 * const b = tf.scalar(5);
 *
 * a.mod(b).print();  // or tf.mod(a, b)
 * ```
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same type as `a`.
 */
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function mod_<T extends Tensor>(a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  let $a = convertToTensor(a, 'a', 'mod');
  let $b = convertToTensor(b, 'b', 'mod');
  [$a, $b] = makeTypesMatch($a, $b);

  const outShape =
      broadcast_util.assertAndGetBroadcastShape($a.shape, $b.shape);
  const der = (dy: Tensor, saved: Tensor[]) => {
    const [$a, $b] = saved;
    const derA = () => {
      const reduceAxes = broadcast_util.getReductionAxes($a.shape, outShape);
      if (reduceAxes.length > 0) {
        return dy.sum(reduceAxes).reshape($a.shape);
      }
      return dy;
    };
    const derB = () => {
      const res = dy.mul($a.div($b).floor().neg());
      const reduceAxes = broadcast_util.getReductionAxes($b.shape, outShape);
      if (reduceAxes.length > 0) {
        return res.sum(reduceAxes).reshape($b.shape);
      }
      return res;
    };
    return {$a: derA, $b: derB};
  };
  return ENGINE.runKernelFunc((backend, save) => {
    const res = backend.mod($a, $b);
    save([$a, $b]);
    return res;
  }, {$a, $b}, der) as T;
}

/**
 * @deprecated
 * Returns the mod of a and b (`a < b ? a : b`) element-wise. Inputs must
 * be the same shape. For broadcasting support, use mod().
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same dtype as `a`.
 */
function modStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');

  const $a = convertToTensor(a, 'a', 'modStrict');
  const $b = convertToTensor(b, 'b', 'modStrict');
  util.assertShapesMatch($a.shape, $b.shape, 'Error in modStrict: ');
  return $a.mod($b);
}

/**
 * @deprecated
 * Returns the min of a and b (`a < b ? a : b`) element-wise. Inputs must
 * be the same shape. For broadcasting support, use minimum().
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same dtype as `a`.
 */
function minimumStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');

  const $a = convertToTensor(a, 'a', 'minimumStrict');
  const $b = convertToTensor(b, 'b', 'minimumStrict');
  util.assertShapesMatch($a.shape, $b.shape, 'Error in minimumStrict: ');
  return $a.minimum($b);
}

/**
 * @deprecated
 * Returns the max of a and b (`a > b ? a : b`) element-wise. Inputs must
 * be the same shape. For broadcasting support, use maximum().
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same dtype as `a`.
 */
function maximumStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');

  const $a = convertToTensor(a, 'a', 'maximumStrict');
  const $b = convertToTensor(b, 'b', 'maximumStrict');
  util.assertShapesMatch($a.shape, $b.shape, 'Error in maximumStrict: ');
  return $a.maximum($b);
}

/**
 * @deprecated
 * Returns (a - b) * (a - b) element-wise.
 *
 * Inputs must be the same shape. For broadcasting support, use
 * `tf.squaredDifference` instead.
 *
 * @param a The first tensor.
 * @param b The second tensor. Must have the same type as `a`.
 */
function squaredDifferenceStrict_<T extends Tensor>(
    a: T|TensorLike, b: T|TensorLike): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');
  const $a = convertToTensor(a, 'a', 'squaredDifferenceStrict');
  const $b = convertToTensor(b, 'b', 'squaredDifferenceStrict');
  util.assertShapesMatch(
      $a.shape, $b.shape, 'Error in squaredDifferenceStrict: ');
  return $a.squaredDifference($b);
}

export const addStrict = op({addStrict_});
export const divStrict = op({divStrict_});
export const maximumStrict = op({maximumStrict_});
export const minimumStrict = op({minimumStrict_});
export const mod = op({mod_});
export const modStrict = op({modStrict_});
export const mul = op({mul_});
export const mulStrict = op({mulStrict_});
export const powStrict = op({powStrict_});
export const squaredDifferenceStrict = op({squaredDifferenceStrict_});
export const subStrict = op({subStrict_});
