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

import {ENGINE} from '../engine';
import {Einsum, EinsumAttrs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';

import {op} from './operation';

/**
 * Tensor contraction over specified indices and outer product.
 *
 * `einsum` allows defining Tensors by defining their element-wise computation.
 * This computation is based on
 * [Einstein summation](https://en.wikipedia.org/wiki/Einstein_notation).
 *
 * Some special cases include:
 *
 * Matrix multiplication:
 * ```js
 * const x = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
 * const y = tf.tensor2d([[0, 1], [2, 3], [4, 5]]);
 * x.print();
 * y.print();
 * tf.einsum('ij,jk->ik', x, y).print();
 * ```
 *
 * Dot product:
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 * const y = tf.tensor1d([0, 1, 2]);
 * x.print();
 * y.print();
 * tf.einsum('i,i->', x, y).print();
 * ```
 *
 * Batch dot product:
 * ```js
 * const x = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
 * const y = tf.tensor2d([[0, 1, 2], [3, 4, 5]]);
 * x.print();
 * y.print();
 * tf.einsum('bi,bi->b', x, y).print();
 * ```
 *
 * Outer prouduct:
 * ```js
 * const x = tf.tensor1d([1, 3, 5]);
 * const y = tf.tensor1d([2, 4, 6]);
 * x.print();
 * y.print();
 * tf.einsum('i,j->ij', x, y).print();
 * ```
 *
 * Matrix transpose:
 * ```js
 * const x = tf.tensor2d([[1, 2], [3, 4]]);
 * x.print();
 * tf.einsum('ij->ji', x).print();
 * ```
 *
 * Batch matrix transpose:
 * ```js
 * const x = tf.tensor3d([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]]);
 * x.print();
 * tf.einsum('bij->bji', x).print();
 * ```
 *
 * Limitations:
 *
 * This implementation of einsum has the following limitations:
 *
 * - Does not support >2 input tensors.
 * - Does not support duplicate axes for any given input tensor. E.g., equation
 *   'ii->' is not suppoted.
 * - The `...` notation is not supported.
 *
 * @param equation a string describing the contraction, in the same format as
 * [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html).
 * @param tensors the input(s) to contract (each one a Tensor), whose shapes
 *     should be consistent with equation.
 * @returns The output tensor.
 *
 * @doc {heading: 'Tensors', subheading: 'Matrices'}
 */
export function einsum_(equation: string, ...tensors: Tensor[]): Tensor {
  const $tensors =
      tensors.map((t, i) => convertToTensor(t, `tensors${i}`, 'einsum'));
  const attrs: EinsumAttrs = {equation};
  return ENGINE.runKernel(
      Einsum, $tensors as {} as NamedTensorMap, attrs as {} as NamedAttrMap);
}

export const einsum = op({einsum_});
