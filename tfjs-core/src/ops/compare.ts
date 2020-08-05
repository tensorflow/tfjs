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
import {assertShapesMatch} from '../util';

import {equal} from './equal';
import {greater} from './greater';
import {greaterEqual} from './greater_equal';
import {less} from './less';
import {lessEqual} from './less_equal';
import {notEqual} from './not_equal';
import {op} from './operation';

/**
 * @deprecated
 * Strict version of `tf.notEqual` that forces `a` and `b` to be of the same
 * shape.
 *
 * @param a The first input tensor.
 * @param b The second input tensor. Must have the same shape and dtype as
 *     `a`.
 */
function notEqualStrict_<T extends Tensor>(
    a: T|TensorLike, b: T|TensorLike): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');
  const $a = convertToTensor(a, 'a', 'notEqualStrict');
  const $b = convertToTensor(b, 'b', 'notEqualStrict');
  assertShapesMatch($a.shape, $b.shape, 'Error in notEqualStrict: ');
  return notEqual($a, $b);
}

/**
 * @deprecated
 * Strict version of `tf.less` that forces `a` and `b` to be of the same
 * shape.
 *
 * @param a The first input tensor.
 * @param b The second input tensor. Must have the same shape and dtype as
 *     `a`.
 */
function lessStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');
  const $a = convertToTensor(a, 'a', 'lessStrict');
  const $b = convertToTensor(b, 'b', 'lessStrict');
  assertShapesMatch($a.shape, $b.shape, 'Error in lessStrict: ');
  return less($a, $b);
}

function equalStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');
  const $a = convertToTensor(a, 'a', 'equalStrict');
  const $b = convertToTensor(b, 'b', 'equalStrict');
  assertShapesMatch($a.shape, $b.shape, 'Error in equalStrict: ');
  return equal($a, $b);
}

function lessEqualStrict_<T extends Tensor>(
    a: T|TensorLike, b: T|TensorLike): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');
  const $a = convertToTensor(a, 'a', 'lessEqualStrict');
  const $b = convertToTensor(b, 'b', 'lessEqualStrict');
  assertShapesMatch($a.shape, $b.shape, 'Error in lessEqualStrict: ');
  return lessEqual($a, $b);
}

function greaterStrict_<T extends Tensor>(a: T|TensorLike, b: T|TensorLike): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');
  const $a = convertToTensor(a, 'a', 'greaterStrict');
  const $b = convertToTensor(b, 'b', 'greaterStrict');
  assertShapesMatch($a.shape, $b.shape, 'Error in greaterStrict: ');
  return greater($a, $b);
}

function greaterEqualStrict_<T extends Tensor>(
    a: T|TensorLike, b: T|TensorLike): T {
  deprecationWarn(
      'strict variants of ops have been deprecated ' +
      'and will be removed in future');
  const $a = convertToTensor(a, 'a', 'greaterEqualStrict');
  const $b = convertToTensor(b, 'b', 'greaterEqualStrict');
  assertShapesMatch($a.shape, $b.shape, 'Error in greaterEqualStrict: ');
  return greaterEqual($a, $b);
}

export const equalStrict = op({equalStrict_});
export const greaterEqualStrict = op({greaterEqualStrict_});
export const greaterStrict = op({greaterStrict_});
export const lessEqualStrict = op({lessEqualStrict_});
export const lessStrict = op({lessStrict_});
export const notEqualStrict = op({notEqualStrict_});
