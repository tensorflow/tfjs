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
import {SparseFillEmptyRows, SparseFillEmptyRowsInputs} from '../../kernel_names';
import {Scalar, Tensor, Tensor1D, Tensor2D} from '../../tensor';
import {NamedTensorMap} from '../../tensor_types';
import {convertToTensor} from '../../tensor_util_env';
import {ScalarLike, TensorLike} from '../../types';
import {op} from '../operation';

/**
 * The input SparseTensor is represented via the map of inputs {`indices`,
 * `values`, `denseShape`}. The output SparseTensor has the same `denseShape`
 * but with indices `outputIndices` and values `outputValues`. This op inserts a
 * single entry for every row that doesn't have any values. The index is created
 * as `[row, 0, ..., 0]` and the inserted value is `defaultValue`.
 *
 * For example, suppose `spInput` has shape [5, 6] and non-empty values:
 * [0, 1]: a
 * [0, 3]: b
 * [2, 0]: c
 * [3, 1]: d
 *
 * Rows 1 and 4 are empty, so the output will be of shape [5, 6] with values:
 * [0, 1]: a
 * [0, 3]: b
 * [1, 0]: `defaultValue`
 * [2, 0]: c
 * [3, 1]: d
 * [4, 0]: `defaultValue`
 *
 * The output SparseTensor will be in row-major order and will have the same
 * shape as the input.
 *
 * This op also returns an indicator vector shaped [dense_shape[0]] such that
 * emptyRowIndicator[i] = True iff row i was an empty row.
 *
 * And a reverse index map vector shaped [indices.shape[0]] that is used during
 * backpropagation, reverseIndexMap[i] = outi s.t. indices[i, j] ==
 * outputIndices[outi, j] for all j
 *
 * ```js
 * const result = tf.sparse.sparseFillEmptyRows(
 *   [[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]],
 *   [0, 10, 13, 14, 32, 33], [5, 6], -1);
 * console.log(result);
 * result['outputIndices'].print(); // [[0, 0], [1, 0], [1, 3], [1, 4],
 *                                  //  [2, 0], [3, 2], [3, 3], [4, 0]]
 * result['outputValues'].print(); // [0, 10, 13, 14,-1, 32, 33, -1]
 * result['emptyRowIndicator'].print(); // [false, false, true, false, true]
 * result['reverseIndexMap'].print(); // [0, 1, 2, 3, 5, 6]
 * ```
 * @param indices: 2-D. the indices of the sparse tensor.
 * @param values: 1-D. the values of the sparse tensor.
 * @param denseShape: 1-D. the shape of the sparse tensor.
 * @param defaultValue: 0-D. default value to insert into location [row, 0, ...,
 *     0] for rows missing from the input sparse tensor.
 * @return A map with the following properties:
 *     - outputIndices
 *     - outputValues: 1-D. the values of the filled sparse tensor.
 *     - emptyRowIndicator: 1-D. whether the dense row was missing in the input
 * sparse tensor.
 *     - reverseIndexMap: 1-D. a map from the input indices to the output
 * indices.
 * @doc {heading: 'Operations', subheading: 'Sparse'}
 */
function sparseFillEmptyRows_(
    indices: Tensor2D|TensorLike, values: Tensor1D|TensorLike,
    denseShape: Tensor1D|TensorLike,
    defaultValue: Scalar|ScalarLike): NamedTensorMap {
  const $indices = convertToTensor(indices, 'indices', 'sparseFillEmptyRows');
  const $values = convertToTensor(values, 'values', 'sparseFillEmptyRows');
  const $denseShape =
      convertToTensor(denseShape, 'denseShape', 'sparseFillEmptyRows');
  const $defaultValue = convertToTensor(
      defaultValue, 'defaultValue', 'sparseFillEmptyRows', $values.dtype);

  if ($indices.rank !== 2) {
    throw new Error(`Indices should be Tensor2D but received shape
        ${$indices.shape}`);
  }
  if ($values.rank !== 1) {
    throw new Error(
        `Values should be Tensor1D but received shape ${$values.shape}`);
  }
  if ($denseShape.rank !== 1) {
    throw new Error(`Dense shape should be Tensor1D but received shape ${
        $denseShape.shape}`);
  }
  if ($defaultValue.rank !== 0) {
    throw new Error(`Default value should be a scalar but received shape ${
        $defaultValue.shape}`);
  }

  const inputs: SparseFillEmptyRowsInputs = {
    indices: $indices,
    values: $values,
    denseShape: $denseShape,
    defaultValue: $defaultValue
  };

  const result: Tensor[] = ENGINE.runKernel(SparseFillEmptyRows, inputs as {});
  return {
    outputIndices: result[0],
    outputValues: result[1],
    emptyRowIndicator: result[2],
    reverseIndexMap: result[3]
  };
}

export const sparseFillEmptyRows = op({sparseFillEmptyRows_});
