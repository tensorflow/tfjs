/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {Rank, TensorLike} from '../types';
import {assertAndGetBroadcastShape} from './broadcast_util';

import {op} from './operation';
import {tensor} from './tensor';

/**
 * Return the shape of shape0 op shape1 with broadcast.
 *
 * compute r0, the broadcasted shape as a tensor.
 * shape0, shape1 and r0 are all integer vectors.
 *
 * This function returns the shape of the result of an operation between
 * two tensors of size shape0 and shape1 performed with broadcast.
 *
 * @param s1 A tensor representing a shape
 * @param s2 A tensor representing a shape
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function broadcastArgs_<R extends Rank>(
    s1: Tensor|TensorLike, s2: Tensor|TensorLike): Tensor<R> {
  const shape1Input = convertToTensor(s1, 'broadcastArgs', 's1', 'int32');
  const shape2Input = convertToTensor(s2, 'broadcastArgs', 's2', 'int32');

  if (shape1Input.rank !== 1) {
    throw new Error(
        `broadcastArgs(): first input must be a vector (rank=1). Has rank ${
            shape1Input.rank}`);
  }

  if (shape2Input.rank !== 1) {
    throw new Error(
        `broadcastArgs(): second input must be a vector (rank=1). Has rank ${
            shape2Input.rank}`);
  }
  return tensor(assertAndGetBroadcastShape(
      shape1Input.arraySync() as number[],
      shape2Input.arraySync() as number[]));
}

export const broadcastArgs = op({broadcastArgs_});
