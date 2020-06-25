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

import {Tensor,} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {matMul} from './mat_mul';
import {op} from './operation';
import {reshape} from './reshape';

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
    const t12D = reshape($t1, [1, -1]);
    const t22D = reshape($t2, [-1, 1]);
    const t1t2 = matMul(t12D, t22D);
    return reshape(t1t2, []);
  } else if ($t1.rank === 1 && $t2.rank === 2) {
    const t12D = reshape($t1, [1, -1]);
    const t22D = reshape($t2, [$t2.shape[0], $t2.shape[1]]);
    const t1t2 = matMul(t12D, t22D);
    return reshape(t1t2, [t1t2.size]);
  } else if ($t1.rank === 2 && $t2.rank === 1) {
    const t22D = reshape($t2, [-1, 1]);
    const t1t2 = matMul($t1, t22D);
    return reshape(t1t2, [t1t2.size]);
  } else {
    const t22D = reshape($t2, [$t2.shape[0], $t2.shape[1]]);
    const t1t2 = matMul($t1, t22D);
    return t1t2;
  }
}

export const dot = op({dot_});
