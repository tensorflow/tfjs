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
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {assertAndGetBroadcastShape} from './broadcast_util';
import {logicalAnd} from './logical_and';
import {logicalNot} from './logical_not';
import {logicalOr} from './logical_or';
import {op} from './operation';

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
  return logicalAnd(logicalOr(a, b), logicalNot(logicalAnd(a, b)));
}

export const logicalXor = op({logicalXor_});
