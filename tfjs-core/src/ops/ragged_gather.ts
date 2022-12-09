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
import {RaggedGather, RaggedGatherAttrs, RaggedGatherInputs} from '../kernel_names';
import {Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {op} from './operation';

/**
 * Gather ragged slices from params axis 0 according to indices.
 *
 * @param paramsNestedSplits: A list of at least 1 Tensor with type 'int32' The
 *     nestedRowSplits tensors that define the row-partitioning for the params
 *     RaggedTensor input.
 * @param paramsDenseValues: A Tensor. The flatValues for the params
 *     RaggedTensor.
 * @param indices: A Tensor. Must be one of type: int32. Indices in the
 *     outermost dimension of params of the values that should be gathered.
 * @param outputRaggedRank: An int that is >= 0. The ragged rank of the output
 *     RaggedTensor. outputNestedSplits will contain this number of rowSplits
 *     tensors. This value should equal indices.shape.ndims + params.raggedRank
 *     - 1.
 * @return A map with the following properties:
 *     - outputNestedSplits: A list of outputRaggedRank Tensor objects with the
 * same type as paramsNestedSplits.
 *     - outputDenseValues: A Tensor. Has the same type as paramsDenseValues.
 * @doc {heading: 'Operations', subheading: 'Ragged'}
 */

interface RaggedGatherMap {
  outputNestedSplits: Tensor[];
  outputDenseValues: Tensor;
}

function raggedGather_(
    paramsNestedSplits: Tensor[], paramsDenseValues: Tensor|TensorLike,
    indices: Tensor|TensorLike, outputRaggedRank: number): RaggedGatherMap {
  const $paramsNestedSplits = paramsNestedSplits.map(
      (t, i) => convertToTensor(t, `tensors${i}`, 'raggedGather', 'int32'));
  const $paramsDenseValues =
      convertToTensor(paramsDenseValues, 'paramsDenseValues', 'raggedGather');
  const $indices = convertToTensor(indices, 'indices', 'raggedGather', 'int32');

  const inputs: RaggedGatherInputs = {
    paramsNestedSplits: $paramsNestedSplits,
    paramsDenseValues: $paramsDenseValues,
    indices: $indices,
  };
  const attrs: RaggedGatherAttrs = {outputRaggedRank};

  const result: Tensor[] =
      ENGINE.runKernel(RaggedGather, inputs as {}, attrs as {});
  return {
    outputNestedSplits: result.slice(0, result.length - 1),
    outputDenseValues: result[result.length - 1],
  };
}

export const raggedGather = /* @__PURE__ */ op({raggedGather_});
