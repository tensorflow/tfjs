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

import {decodeEinsumEquation, getEinsumPermutation, isIdentityPermutation, transformEinsumInput} from '../backends/einsum_util';
import {ENGINE} from '../engine';
import {dispose} from '../globals';
import {Einsum, EinsumAttrs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {matMul} from './mat_mul';

import {op} from './operation';
import {reshape} from './reshape';
import {transpose} from './transpose';

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
 *   'ii->' is not supported.
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

  if ($tensors.length === 2) {
    const {allDims, summedDims, idDims} = decodeEinsumEquation(equation, 2);
    if (summedDims.length === 1) {
      // If the einsum has two operands and has one dimension to be summed, it
      // could be reduced as a BatchMatMul.
      // The dimensions of each operand could be divided into three categories:
      // shared dimensions (both the two operands have), summed dimension (both
      // the two operands have and it's the dimension to sum on), distinguished
      // dimensions (an operand exclusively has), which are corresponding to
      // Batch, K and M(N) dimensions of BatchMatMul. The einsum could be
      // reduced as BatchMatMul with the following steps:
      //   1. Transpose the two operands' dimensions to the order of
      //   [...sharedDimensions, ...distinguishedDimensions, summed dimension].
      //   2. Reshape the first operands' shape as [sharedDimensionsProduct,
      //   distinguishedDimensionsProduct, summedDim], while reshape and
      //   transpose the second operands' shape as [sharedDimensionsProduct,
      //   summedDim distinguishedDimensionsProduct], Which are corresponding to
      //   [Batch, M, K] X [Batch, K, N].
      //   3. Compute BatchMatMul.
      //   4. Transpose and reshape the result [Batch, M, N] to the target
      //   shape.
      const summedDim = summedDims[0];

      const sharedDims: number[] = [];
      const outputShape: number[] = new Array<number>(allDims.length - 1);
      outputShape.fill(-1);
      const distinguishedDimsA: number[] = [];
      const distinguishedDimsB: number[] = [];

      for (let i = 0; i < $tensors[0].rank; i++) {
        const dim = idDims[0][i];
        if (dim === summedDim) {
          continue;
        } else if (idDims[1].indexOf(dim) !== -1) {
          sharedDims.push(dim);
        } else {
          distinguishedDimsA.push(dim);
        }
        outputShape[dim] = $tensors[0].shape[i];
      }

      let isReducible = true;
      for (let i = 0; i < $tensors[1].rank; i++) {
        const dim = idDims[1][i];
        const dimSize = $tensors[1].shape[i];
        if (dim === summedDim) {
          continue;
        } else if (idDims[1].indexOf(dim) !== -1) {
          sharedDims.push(dim);
          if (outputShape[dim] !== dimSize) {
            isReducible = false;
            break;
          }
        } else {
          distinguishedDimsB.push(dim);
          outputShape[dim] = dimSize;
        }
      }

      if (isReducible) {
        const intermediates = [];
        const {resultTensor: tensorA, isSumDimLast: aSumDimLast} =
            transformEinsumInput($tensors[0], idDims[0], sharedDims, summedDim);
        const {resultTensor: tensorB, isSumDimLast: bSumDimLast} =
            transformEinsumInput($tensors[1], idDims[1], sharedDims, summedDim);
        intermediates.push(tensorA);
        intermediates.push(tensorB);

        let res = matMul(tensorA, tensorB, !aSumDimLast, bSumDimLast);
        const outputIdDims =
            sharedDims.concat(distinguishedDimsA).concat(distinguishedDimsB);
        const outputShape: number[] = outputIdDims.map(dim => outputShape[dim]);
        intermediates.push(res);
        res = reshape(res, outputShape);
        if (!isIdentityPermutation(outputIdDims)) {
          const {permutationIndices} =
              getEinsumPermutation(outputIdDims.length, outputIdDims);
          intermediates.push(res);
          res = transpose(res, permutationIndices);
        }

        for (const tensor of intermediates) {
          dispose(tensor);
        }
        return res;
      }
    }
  }


  const attrs: EinsumAttrs = {equation};
  return ENGINE.runKernel(
      Einsum, $tensors as unknown as NamedTensorMap,
      attrs as unknown as NamedAttrMap);
}

export const einsum = /* @__PURE__ */ op({einsum_});
