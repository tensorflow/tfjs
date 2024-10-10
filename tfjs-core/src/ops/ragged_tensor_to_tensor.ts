/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import {RaggedTensorToTensor, RaggedTensorToTensorAttrs, RaggedTensorToTensorInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {op} from './operation';

/**
 * Create a dense tensor from a ragged tensor, possibly altering its shape.
 *
 * The raggedTensorToTensor op creates a dense tensor from am array of row
 * partition tensors, a value vector, and default values. If the shape is
 * unspecified, the minimal shape required to contain all the elements in the
 * ragged tensor (the natural shape) will be used. If some dimensions are left
 * unspecified, then the size of the natural shape is used in that dimension.
 *
 * The defaultValue will be broadcast to the output shape. After that, the
 * values from the ragged tensor overwrite the default values. Note that the
 * defaultValue must have less dimensions than the value.
 *
 * The row partition tensors are in the order of the dimensions. At present, the
 * types can be: "ROW_SPLITS": the row_splits tensor from the ragged tensor.
 *   "VALUE_ROWIDS": the value_rowids tensor from the ragged tensor.
 *   "FIRST_DIM_SIZE": if value_rowids is used for the first dimension, then it
 * is preceded by "FIRST_DIM_SIZE".
 * ```
 * @param shape: A Tensor. Must be one of the following types: 'int32'. The
 *     desired shape of the output tensor. If left unspecified (empty), the
 *     minimal shape required to contain all the elements in the ragged tensor
 *     (the natural shape) will be used. If some dimensions are left
 *     unspecified, then the size of the natural shape is used in that
 *     dimension.
 *
 *     Note that dense dimensions cannot be modified by the shape argument.
 *     Trying to change the size of a dense dimension will cause the op to fail.
 *     Examples: natural shape: [4, 5, 6] shape: -1 output shape: [4, 5, 6]
 *
 *     natural shape: [4, 5, 6] shape: [3, -1, 2] output shape: [3, 5, 2]
 *
 *     natural shape: [4, 5, 6] shape: [3, 7, 2] output shape: [3, 7, 2]
 * @param values: A Tensor. A 1D tensor representing the values of the ragged
 *     tensor.
 * @param defaultValue: A Tensor. Must have the same type as values. The
 *     defaultValue when the shape is larger than the ragged tensor. The
 *     defaultValue is broadcast until it is the shape of the output tensor,
 *     and then overwritten by values in the ragged tensor. The default value
 *     must be compatible with this broadcast operation, and must have fewer
 *     dimensions than the value tensor.
 * @param rowPartitionTensors: A list of at least 1 Tensor objects with the same
 *     type in: 'int32'.
 * @param rowPartitionTypes: A list of strings. The types of the row partition
 *     tensors. At present, these can be:
 *     "ROW_SPLITS": the row_splits tensor from the ragged tensor.
 *     "VALUE_ROWIDS": the value_rowids tensor from the ragged tensor.
 *     "FIRST_DIM_SIZE": if value_rowids is used for the first dimension, then
 *         it is preceded by "FIRST_DIM_SIZE". The tensors are in the order of
 *         the dimensions.
 * @return A Tensor. Has the same type as values.
 * @doc {heading: 'Operations', subheading: 'Ragged'}
 */
function raggedTensorToTensor_(
    shape: Tensor|TensorLike, values: Tensor|TensorLike,
    defaultValue: Tensor|TensorLike, rowPartitionTensors: Tensor[],
    rowPartitionTypes: string[]): Tensor {
  const $shape =
      convertToTensor(shape, 'shape', 'raggedTensorToTensor', 'int32');
  const $values = convertToTensor(values, 'values', 'raggedTensorToTensor');
  const $defaultValue = convertToTensor(
      defaultValue, 'defaultValue', 'raggedTensorToTensor', $values.dtype);
  const $rowPartitionTensors = rowPartitionTensors.map(
      (t, i) =>
          convertToTensor(t, `tensors${i}`, 'raggedTensorToTensor', 'int32'));

  const inputs: RaggedTensorToTensorInputs = {
    shape: $shape,
    values: $values,
    defaultValue: $defaultValue,
    rowPartitionTensors: $rowPartitionTensors
  };
  const attrs: RaggedTensorToTensorAttrs = {rowPartitionTypes};

  return ENGINE.runKernel(RaggedTensorToTensor, inputs as {}, attrs as {});
}

export const raggedTensorToTensor = /* @__PURE__ */ op({raggedTensorToTensor_});
