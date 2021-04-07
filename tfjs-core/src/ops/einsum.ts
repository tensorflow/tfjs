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

import {checkDimSizes, decodeEquation, getComputePath, getPermutation} from '../backends/einsum_util';
import {ENGINE} from '../engine';
import {Multiply, Reshape, Sum, Transpose} from '../kernel_names';
import {Tensor} from '../tensor';
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
 * Limitations:
 *
 * This implementation of einsum has the following limitations:
 *
 * - Does not support >2 input tensors.
 * - Does not support duplicate axes for any given input tensor. E.g., equation
 *   'ii->' is not suppoted.
 * - For two or more input tensors, up to only one summation axis is supported.
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
  const {allDims, summedDims, idDims} =
      decodeEquation(equation, tensors.length);
  checkDimSizes(allDims.length, idDims, tensors);
  const {path, steps} = getComputePath(summedDims, idDims);

  const nSteps = steps.length;
  let out: Tensor|null = null;
  let numDimsRemaining = allDims.length;
  for (let i = 0; i < nSteps; ++i) {
    for (const idTerm of steps[i]) {
      const {permutationIndices, expandDims: dimsToExpand} =
          getPermutation(numDimsRemaining, idDims[idTerm]);
      // tslint:disable-next-line:no-unnecessary-type-assertion
      let x = ENGINE.runKernel(
                  Transpose, {x: tensors[idTerm]},
                  {perm: permutationIndices}) as Tensor;
      const targetShape: number[] = x.shape;
      for (let k = 0; k < dimsToExpand.length; ++k) {
        targetShape.splice(dimsToExpand[k], 0, 1);
      }

      x = ENGINE.runKernel(Reshape, {x}, {shape: targetShape});
      out = out === null ? x : ENGINE.runKernel(Multiply, {a: out, b: x});
    }
    if (i < nSteps - 1) {
      if (path[i] >= 0) {
        out = ENGINE.runKernel(
            Sum, {x: out},
            {axis: path[i] < out.shape.length ? path[i] : undefined});
      }
      numDimsRemaining--;
    }
  }
  return out;
}

export const einsum = op({einsum_});
