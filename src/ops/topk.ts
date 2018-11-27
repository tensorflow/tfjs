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

import {ENV} from '../environment';
import {NumericTensor, Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {op} from './operation';

/**
 * Finds the values and indices of the `k` largest entries along the last
 * dimension.
 *
 * If the input is a vector (rank=1), finds the k largest entries in the vector
 * and outputs their values and indices as vectors. Thus values[j] is the j-th
 * largest entry in input, and its index is indices[j].
 * For higher rank inputs, computes the top k entries along the last dimension.
 *
 * If two elements are equal, the lower-index element appears first.
 *
 * ```js
 * const a = tf.tensor2d([[1, 5], [4, 3]]);
 * const {values, indices} = tf.topk(a);
 * values.print();
 * indices.print();
 * ```
 * @param x 1-D or higher `tf.Tensor` with last dimension being at least `k`.
 * @param k Number of top elements to look for along the last dimension.
 * @param sorted If true, the resulting `k` elements will be sorted by the
 *     values in descending order.
 */
/** @doc {heading: 'Operations', subheading: 'Evaluation'} */
function topk_<T extends Tensor>(
    x: T|TensorLike, k = 1, sorted = true): {values: T, indices: T} {
  const $x = convertToTensor(x, 'x', 'topk');
  if ($x.rank === 0) {
    throw new Error('topk() expects the input to be of rank 1 or higher');
  }
  const lastDim = $x.shape[$x.shape.length - 1];
  if (k > lastDim) {
    throw new Error(
        `'k' passed to topk() must be <= the last dimension (${lastDim}) ` +
        `but got ${k}`);
  }

  const [values, indices] =
      ENV.engine.runKernel(b => b.topk($x as NumericTensor, k, sorted), {$x});
  return {values, indices} as {values: T, indices: T};
}

export const topk = op({topk_});
