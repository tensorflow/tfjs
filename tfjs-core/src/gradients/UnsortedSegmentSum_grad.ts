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

import {UnsortedSegmentSum} from '../kernel_names';
import {GradConfig} from '../kernel_registry';
import {expandDims} from '../ops/expand_dims';
import {gather} from '../ops/gather';
import {greaterEqual} from '../ops/greater_equal';
import {logicalAnd} from '../ops/logical_and';
import {maximum} from '../ops/maximum';
import {ones} from '../ops/ones';
import {scalar} from '../ops/scalar';
import {where} from '../ops/where';
import {zerosLike} from '../ops/zeros_like';
import {Tensor, Tensor1D} from '../tensor';

export const unsortedSegmentSumGradConfig: GradConfig = {
  kernelName: UnsortedSegmentSum,
  inputsToSave: ['segmentIds'],
  gradFunc: (dy: Tensor, saved: Tensor[]) => {
    const [segmentIds] = saved;

    const derX = () => {
      return gatherDropNegatives(dy, segmentIds as Tensor1D);
    };
    return {x: derX};
  }
};

function gatherDropNegatives<T extends Tensor>(x: T, indices: Tensor1D) {
  // Helper function for unsorted segment ops. Gathers params for
  // positive segment ids and gathers 0 for inputs with negative segment id.
  // Mirrors _GatherDropNegatives from tensorflow/python/ops/math_grad.py
  const zeroClippedIndices = maximum(indices, zerosLike(indices));
  const gathered = gather(x, zeroClippedIndices as Tensor1D);
  let isPositive = greaterEqual(indices, scalar(0, 'int32'));
  const numIters = gathered.rank - isPositive.rank;
  for (let i = 0; i < numIters; ++i) {
    isPositive = expandDims(isPositive, i + 1);
  }
  isPositive = logicalAnd(isPositive, ones(gathered.shape, 'bool'));
  const zeroSlice = zerosLike(gathered);
  return where(isPositive, gathered, zeroSlice);
}
