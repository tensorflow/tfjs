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

import {op} from './operation';
import {pad} from './pad';
import {tensor} from './tensor';
import {zeros} from './zeros';

/**
 * Return the shape of shape0 op shape1 with broadcast.
 *
 * compute r0, the broadcasted shape as a tensor.
 * shape0, shape1 and r0 are all integer vectors.
 *
 * This function returns the shape of the result of an operation between
 * two tensors of size shape0 and shape1 performed with broadcast.
 *
 * @param shape1 A tensor representing a shape
 * @param shape2 A tensor representing a shape
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function broadcastArgs_<R extends Rank>(
    shape1: Tensor|TensorLike, shape2: Tensor|TensorLike): Tensor<R> {
  const shape1Input = convertToTensor(shape1, 'broadcastArgs', 's1', 'int32');
  const shape2Input = convertToTensor(shape2, 'broadcastArgs', 's2', 'int32');

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

  // Pad with 1s.
  const maxRank = Math.max(shape1Input.shape[0], shape2Input.shape[0]);
  const shape1Padded =
      pad(shape1Input, [[maxRank - shape1Input.shape[0], 0]], 1).arraySync() as
      number[];
  const shape2Padded =
      pad(shape2Input, [[maxRank - shape2Input.shape[0], 0]], 1).arraySync() as
      number[];

  const output = zeros([maxRank]).arraySync() as number[];
  // Going through each dimension starting from the outer-most
  // dimension, compares dimension of the padded
  // shape1 and shape2. They are compatible if they are equal or either is 1.
  for (let i = 0; i < maxRank; ++i) {
    if (shape1Padded[i] === 1) {
      output[i] = shape2Padded[i];
    } else if (shape2Padded[i] === 1) {
      output[i] = shape1Padded[i];
    } else if (shape1Padded[i] === shape2Padded[i]) {
      output[i] = shape1Padded[i];
    } else {
      throw new Error(`broadcastArgs(): inputs are incompatible.`);
    }
  }

  return tensor(output);
}

export const broadcastArgs = op({broadcastArgs_});
