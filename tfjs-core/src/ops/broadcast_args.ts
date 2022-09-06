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

import { NamedTensorMap } from '../tensor_types';
import { ENGINE } from '../engine';
import { BroadcastArgs, BroadcastArgsInputs } from '../kernel_names';
import { Tensor } from '../tensor';
import { convertToTensor } from '../tensor_util_env';
import { Rank, TensorLike } from '../types';

import { op } from './operation';

/**
 * Return the shape of s0 op s1 with broadcast.
 *
 * compute r0, the broadcasted shape as a tensor.
 * s0, s1 and r0 are all integer vectors.
 *
 * This function returns the shape of the result of an operation between
 * two tensors of size s0 and s1 performed with broadcast.
 *
 * @param s0 A tensor representing a shape
 * @param s1 A tensor representing a shape
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function broadcastArgs_<R extends Rank>(
  s0: Tensor | TensorLike, s1: Tensor | TensorLike): Tensor<R> {
  const shape1Input = convertToTensor(s0, 's0', 'broadcastArgs', 'int32');
  const shape2Input = convertToTensor(s1, 's1', 'broadcastArgs', 'int32');

  if (shape1Input.rank !== 1) {
    throw new Error(
      'broadcastArgs(): first input must be a vector (rank=1). ' +
      `Has rank ${shape1Input.rank}`);
  }

  if (shape2Input.rank !== 1) {
    throw new Error(
      'broadcastArgs(): second input must be a vector (rank=1). ' +
      `Has rank ${shape2Input.rank}`);
  }

  const inputs: BroadcastArgsInputs = { s0: shape1Input, s1: shape2Input };
  return ENGINE.runKernel(BroadcastArgs, inputs as {} as NamedTensorMap);
}

export const broadcastArgs = op({ broadcastArgs_ });
