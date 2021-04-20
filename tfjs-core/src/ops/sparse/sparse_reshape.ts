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

import {ENGINE} from '../../engine';
import {SparseReshape, SparseReshapeInputs} from '../../kernel_names';
import {Tensor, Tensor1D, Tensor2D} from '../../tensor';
import {NamedTensorMap} from '../../tensor_types';
import {convertToTensor} from '../../tensor_util_env';
import {TensorLike} from '../../types';
import {op} from '../operation';

/**
 * This operation has the same semantics as reshape on the represented dense
 * tensor. The `inputIndices` are recomputed based on the requested `newShape`.
 * If one component of `newShape` is the special value -1, the size of that
 * dimension is computed so that the total dense size remains constant. At most
 * one component of `newShape` can be -1. The number of dense elements implied
 * by `newShape` must be the same as the number of dense elements originally
 * implied by `inputShape`. Reshaping does not affect the order of values in the
 * SparseTensor. If the input tensor has rank R_in and N non-empty values, and
 * `newShape` has length R_out, then `inputIndices` has shape [N, R_in],
 * `inputShape` has length R_in, `outputIndices` has shape [N, R_out], and
 * `outputShape` has length R_out.
 *
 * ```js
 * const result = tf.sparse.sparseReshape(
 *   [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 2, 3]],
 *   [2, 3, 6], [9, -1]);
 * console.log(result);
 * result['outputIndices'].print(); //[[0, 0], [0, 1], [1, 2], [4, 2], [8, 1]]
 * result['outputShape'].print(); // [9, 4]
 * ```
 * @param inputIndices: 2-D. N x R_in matrix with the indices of non-empty
 * values in a SparseTensor.
 * @param inputShape: 1-D. R_in Tensor1D with the input SparseTensor's dense
 * shape.
 * @param newShape: 1-D. R_out Tensor1D with the requested new dense shape.
 * @return A map with the following properties:
 *     - outputIndices: 2-D. N x R_out matrix with the updated indices of
 *       non-empty values in the output SparseTensor.
 *     - outputShape: 1-D. R_out vector with the full dense shape of the output
 *       SparseTensor. This is the same as newShape but with any -1 dimensions
 *        filled in.
 * @doc {heading: 'Operations', subheading: 'Sparse'}
 */
function sparseReshape_(
    inputIndices: Tensor2D|TensorLike, inputShape: Tensor1D|TensorLike,
    newShape: Tensor1D|TensorLike): NamedTensorMap {
  const $inputIndices =
      convertToTensor(inputIndices, 'inputIndices', 'sparseReshape');
  const $inputShape =
      convertToTensor(inputShape, 'inputShape', 'sparseReshape');
  const $newShape = convertToTensor(newShape, 'newShape', 'sparseReshape');

  if ($inputIndices.rank !== 2) {
    throw new Error(`Input indices should be Tensor2D but received shape
        ${$inputIndices.shape}`);
  }
  if ($inputShape.rank !== 1) {
    throw new Error(`Input shape should be Tensor1D but received shape ${
        $inputShape.shape}`);
  }
  if ($newShape.rank !== 1) {
    throw new Error(
        `New shape should be Tensor1D but received shape ${$newShape.shape}`);
  }

  const inputs: SparseReshapeInputs = {
    inputIndices: $inputIndices,
    inputShape: $inputShape,
    newShape: $newShape
  };
  const result: Tensor[] = ENGINE.runKernel(SparseReshape, inputs as {});
  return {outputIndices: result[0], outputShape: result[1]};
}

export const sparseReshape = op({sparseReshape_});
