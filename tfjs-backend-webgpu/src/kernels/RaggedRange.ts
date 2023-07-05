/**
 * @license
 * Copyright 2023 Google LLC.
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

import {KernelConfig, KernelFunc, RaggedRange, RaggedRangeInputs, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {prefixSum} from '../kernel_utils/prefix_sum';
import {RangeDenseValuesProgram, RangeSizeProgram} from '../ragged_range_webgpu';

export function raggedRange(
    args: {inputs: RaggedRangeInputs, backend: WebGPUBackend}):
    [TensorInfo, TensorInfo] {
  const {inputs, backend} = args;
  const {starts, limits, deltas} = inputs;

  const startsShape = starts.shape;
  const limitsShape = limits.shape;
  const deltasShape = deltas.shape;

  // Check input tensor shapes.
  if (startsShape.length > 1) {
    throw new Error('starts must be a scalar or vector');
  }
  if (limitsShape.length > 1) {
    throw new Error('limits must be a scalar or vector');
  }
  if (deltasShape.length > 1) {
    throw new Error('deltas must be a scalar or vector');
  }

  // Determine which tensors we need to broadcast.
  const broadcastStarts = startsShape.length === 0;
  const broadcastLimits = limitsShape.length === 0;
  const broadcastDeltas = deltasShape.length === 0;

  // nRows (number of output rows) is the size of the non-broadcast inputs,
  // or 1 if all inputs are scalars.
  const inSizes: number[] = [];
  if (!broadcastStarts) {
    inSizes.push(startsShape[0]);
  }
  if (!broadcastLimits) {
    inSizes.push(limitsShape[0]);
  }
  if (!broadcastDeltas) {
    inSizes.push(deltasShape[0]);
  }

  for (let i = 1; i < inSizes.length; ++i) {
    if (inSizes[i] !== inSizes[i - 1]) {
      throw new Error('starts, limits, and deltas must have the same shape');
    }
  }
  const nRows = inSizes.length === 0 ? 1 : inSizes[0];

  let program: RangeSizeProgram|RangeDenseValuesProgram;
  program = new RangeSizeProgram(nRows);
  const rangeSize =
      backend.runWebGPUProgram(program, [starts, limits, deltas], 'int32');
  const rtNestedSplits = prefixSum(rangeSize, backend);

  const $rtNestedSplits = backend.readSync(rtNestedSplits.dataId) as TypedArray;
  const nVals = $rtNestedSplits[nRows];
  program = new RangeDenseValuesProgram(nVals);
  const uniformData = [{type: 'int32', data: [nRows + 1]}];
  const rtDenseValues = backend.runWebGPUProgram(
      program, [starts, deltas, rtNestedSplits], starts.dtype, uniformData);

  backend.disposeData(rangeSize.dataId);
  return [rtNestedSplits, rtDenseValues];
}

export const raggedRangeConfig: KernelConfig = {
  kernelName: RaggedRange,
  backendName: 'webgpu',
  kernelFunc: raggedRange as unknown as KernelFunc,
};
