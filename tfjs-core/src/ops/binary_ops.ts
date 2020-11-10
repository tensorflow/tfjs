/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {deprecationWarn} from '../globals';
import {Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {add} from './add';
import {div} from './div';
import {maximum} from './maximum';
import {minimum} from './minimum';
import {mod} from './mod';
import {mul} from './mul';
import {op} from './operation';
import {pow} from './pow';
import {squaredDifference} from './squared_difference';
import {sub} from './sub';

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
  return add($a, $b);
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
  return sub($a, $b);
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
  return pow(base, exp);
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
  return mul($a, $b);
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
  return div($a, $b);
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
  return mod($a, $b);
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
  return minimum($a, $b);
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
  return maximum($a, $b);
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
  return squaredDifference($a, $b);
}

export const addStrict = op({addStrict_});
export const divStrict = op({divStrict_});
export const maximumStrict = op({maximumStrict_});
export const minimumStrict = op({minimumStrict_});
export const modStrict = op({modStrict_});
export const mulStrict = op({mulStrict_});
export const powStrict = op({powStrict_});
export const squaredDifferenceStrict = op({squaredDifferenceStrict_});
export const subStrict = op({subStrict_});
