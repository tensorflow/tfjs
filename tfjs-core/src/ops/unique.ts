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

import {ENGINE} from '../engine';
import {Unique, UniqueInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {op} from './operation';

/**
 * Finds the unique elements in a 1-D tensor.
 *
 * It returns a tensor `values` containing all of the unique elements along the
 * `axis` of the given tensor `x` in the same order that they occur along the
 * `axis` in `x`; `x` does not need to be sorted. It also returns a tensor
 * `indices` the same size as the number of the elements in `x` along the `axis`
 * dimension. It contains the index in the unique output `values`.
 *
 * For now, only 1-D tensor is support, and the `axis` parameter is not used.
 * Tensors with higher dimensions will be supported in UniqueV2.
 *
 * ```js
 * const a = tf.tensor2d([1, 1, 2, 4, 4, 4, 7, 8, 8]);
 * const {values, indices} = tf.unique(a);
 * values.print();
 * indices.print();
 * ```
 * @param x 1-D or higher `tf.Tensor`.
 * @param axis The axis of the tensor to find the unique elements.
 *
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
function unique_<T extends Tensor>(
    x: T|TensorLike, axis?: number): {values: T, indices: T} {
  // x can be of any dtype, thus null as the last argument.
  const $x = convertToTensor(x, 'x', 'unique', null);
  const inputs: UniqueInputs = {x: $x};
  const [values, indices] =
      ENGINE.runKernel(
          Unique, inputs as {} as NamedTensorMap, {} /* attrs */) as Tensor[];
  return {values, indices} as {values: T, indices: T};
}

export const unique = op({unique_});
