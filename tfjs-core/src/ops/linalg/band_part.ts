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

import {Tensor} from '../../tensor';
import {convertToTensor} from '../../tensor_util_env';
import {TensorLike} from '../../types';
import {assert} from '../../util';

import {greaterEqual} from '../greater_equal';
import {lessEqual} from '../less_equal';
import {logicalAnd} from '../logical_and';
import {op} from '../operation';
import {range} from '../range';
import {reshape} from '../reshape';
import {scalar} from '../scalar';
import {stack} from '../stack';
import {sub} from '../sub';
import {unstack} from '../unstack';
import {where} from '../where';
import {zeros} from '../zeros';

/**
 * Copy a tensor setting everything outside a central band in each innermost
 * matrix to zero.
 *
 * The band part is computed as follows: Assume input has `k` dimensions
 * `[I, J, K, ..., M, N]`, then the output is a tensor with the same shape where
 * `band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.
 * The indicator function
 * `in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower))`
 * `&& (num_upper < 0 || (n-m) <= num_upper)`
 *
 * ```js
 * const x = tf.tensor2d([[ 0,  1,  2, 3],
 *                        [-1,  0,  1, 2],
 *                        [-2, -1,  0, 1],
 *                        [-3, -2, -1, 0]]);
 * let y = tf.linalg.bandPart(x, 1, -1);
 * y.print(); // [[ 0,  1,  2, 3],
 *            //  [-1,  0,  1, 2],
 *            //  [ 0, -1,  0, 1],
 *            //  [ 0, 0 , -1, 0]]
 * let z = tf.linalg.bandPart(x, 2, 1);
 * z.print(); // [[ 0,  1,  0, 0],
 *            //  [-1,  0,  1, 0],
 *            //  [-2, -1,  0, 1],
 *            //  [ 0, -2, -1, 0]]
 * ```
 *
 * @param x Rank `k` tensor
 * @param numLower Number of subdiagonals to keep.
 *   If negative, keep entire lower triangle.
 * @param numUpper Number of subdiagonals to keep.
 *   If negative, keep entire upper triangle.
 * @returns Rank `k` tensor of the same shape as input.
 *   The extracted banded tensor.
 *
 * @doc {heading:'Operations', subheading:'Linear Algebra', namespace:'linalg'}
 */
function bandPart_<T extends Tensor>(
    a: T|TensorLike, numLower: number, numUpper: number): T {
  assert(
      numLower % 1 === 0,
      () => `bandPart(): numLower must be an integer, got ${numLower}.`);
  assert(
      numUpper % 1 === 0,
      () => `bandPart(): numUpper must be an integer, got ${numUpper}.`);

  const $a = convertToTensor(a, 'a', 'bandPart');

  assert(
      $a.rank >= 2,
      () => `bandPart(): Rank must be at least 2, got ${$a.rank}.`);

  const shape = $a.shape;
  const [M, N] = $a.shape.slice(-2);

  if (!(numLower <= M)) {
    throw new Error(
        `bandPart(): numLower (${numLower})` +
        ` must not be greater than the number of rows (${M}).`);
  }
  if (!(numUpper <= N)) {
    throw new Error(
        `bandPart(): numUpper (${numUpper})` +
        ` must not be greater than the number of columns (${N}).`);
  }

  if (numLower < 0) {
    numLower = M;
  }
  if (numUpper < 0) {
    numUpper = N;
  }

  const i = reshape(range(0, M, 1, 'int32'), [-1, 1]);
  const j = range(0, N, 1, 'int32');
  const ij = sub(i, j);

  const inBand = logicalAnd(
      lessEqual(ij, scalar(+numLower, 'int32')),
      greaterEqual(ij, scalar(-numUpper, 'int32')));

  const zero = zeros([M, N], $a.dtype);

  return reshape(
             stack(unstack(reshape($a, [-1, M, N]))
                       .map(mat => where(inBand, mat, zero))),
             shape) as T;
}

export const bandPart = op({bandPart_});
