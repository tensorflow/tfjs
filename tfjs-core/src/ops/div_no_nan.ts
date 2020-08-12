/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {Tensor} from '../tensor';
import {makeTypesMatch} from '../tensor_util';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {div} from './div';
import {equal} from './equal';
import {op} from './operation';
import {where} from './where';
import {zerosLike} from './zeros_like';

/**
 * Divides two `tf.Tensor`s element-wise, A / B. Supports broadcasting. Return 0
 * if denominator is 0.
 *
 *
 * ```js
 * const a = tf.tensor1d([1, 4, 9, 16]);
 * const b = tf.tensor1d([1, 2, 3, 4]);
 * const c = tf.tensor1d([0, 0, 0, 0]);
 *
 * a.divNoNan(b).print();  // or tf.divNoNan(a, b)
 * a.divNoNan(c).print();  // or tf.divNoNan(a, c)
 * ```
 *
 * ```js
 * // Broadcast div a with b.
 * const a = tf.tensor1d([2, 4, 6, 8]);
 * const b = tf.scalar(2);
 * const c = tf.scalar(0);
 *
 * a.divNoNan(b).print();  // or tf.divNoNan(a, b)
 * a.divNoNan(c).print();  // or tf.divNoNan(a, c)
 * ```
 *
 * @param a The first tensor as the numerator.
 * @param b The second tensor as the denominator. Must have the same dtype as
 * `a`.
 */
/** @doc {heading: 'Operations', subheading: 'Arithmetic'} */
function divNoNan_<T extends Tensor>(
    a: Tensor|TensorLike, b: Tensor|TensorLike): T {
  // TODO: Make this into its own kernel.
  let $a = convertToTensor(a, 'a', 'div');
  let $b = convertToTensor(b, 'b', 'div');
  [$a, $b] = makeTypesMatch($a, $b);

  const divResult = div($a, $b);
  const zeros = zerosLike(divResult);
  const bEqualsZero = equal($b, zeros);
  return where(bEqualsZero, zeros, divResult) as T;
}

export const divNoNan = op({divNoNan_});
