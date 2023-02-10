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

import {Scalar, Tensor} from '../../tensor';
import {convertToTensor} from '../../tensor_util_env';
import {TensorLike} from '../../types';
import {assert} from '../../util';

import {greaterEqual} from '../greater_equal';
import {less} from '../less';
import {lessEqual} from '../less_equal';
import {logicalAnd} from '../logical_and';
import {minimum} from '../minimum';
import {neg} from '../neg';
import {op} from '../operation';
import {range} from '../range';
import {reshape} from '../reshape';
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
 * `in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)`
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
    a: T|TensorLike, numLower: number|Scalar, numUpper: number|Scalar): T {
  const $a = convertToTensor(a, 'a', 'bandPart');
  assert(
      $a.rank >= 2,
      () => `bandPart(): Rank must be at least 2, got ${$a.rank}.`);

  const shape = $a.shape;
  const [M, N] = $a.shape.slice(-2);

  let $numLower: Scalar;
  let $numUpper: Scalar;
  if (typeof numLower === 'number') {
    assert(
        numLower % 1 === 0,
        () => `bandPart(): numLower must be an integer, got ${numLower}.`);
    assert(
        numLower <= M,
        () => `bandPart(): numLower (${numLower})` +
            ` must not be greater than the number of rows (${M}).`);
    $numLower =
        convertToTensor(numLower < 0 ? M : numLower, 'numLower', 'bandPart') as
        Scalar;
  } else {
    assert(
        numLower.dtype === 'int32',
        () => `bandPart(): numLower's dtype must be an int32.`);
    // If numLower is a Scalar, checking `numLower <= M` could hurt performance,
    // but minimum(numLower, M) could avoid unexpected results.
    $numLower = where(less(numLower, 0), M, minimum(numLower, M)) as Scalar;
  }

  if (typeof numUpper === 'number') {
    assert(
        numUpper % 1 === 0,
        () => `bandPart(): numUpper must be an integer, got ${numUpper}.`);
    assert(
        numUpper <= N,
        () => `bandPart(): numUpper (${numUpper})` +
            ` must not be greater than the number of columns (${N}).`);
    $numUpper =
        convertToTensor(numUpper < 0 ? N : numUpper, 'numUpper', 'bandPart') as
        Scalar;
  } else {
    assert(
        numUpper.dtype === 'int32',
        () => `bandPart(): numUpper's dtype must be an int32.`);
    $numUpper = where(less(numUpper, 0), N, minimum(numUpper, N)) as Scalar;
  }

  const i = reshape(range(0, M, 1, 'int32'), [-1, 1]);
  const j = range(0, N, 1, 'int32');
  const ij = sub(i, j);

  const inBand =
      logicalAnd(lessEqual(ij, $numLower), greaterEqual(ij, neg($numUpper)));

  const zero = zeros([M, N], $a.dtype);

  return reshape(
             stack(unstack(reshape($a, [-1, M, N]))
                       .map(mat => where(inBand, mat, zero))),
             shape) as T;
}

export const bandPart = /* @__PURE__ */ op({bandPart_});
