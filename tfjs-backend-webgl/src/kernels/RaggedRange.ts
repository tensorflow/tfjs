/**
 * @license
 * Copyright 2022 Google LLC.
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

import {MathBackendWebGL} from '../backend_webgl';
import {raggedRangeImplCPU} from '../kernel_utils/shared';

export function raggedRange(
    args: {inputs: RaggedRangeInputs, backend: MathBackendWebGL}):
    [TensorInfo, TensorInfo] {
  const {inputs, backend} = args;
  const {starts, limits, deltas} = inputs;

  const $starts = backend.readSync(starts.dataId) as TypedArray;
  const $limits = backend.readSync(limits.dataId) as TypedArray;
  const $deltas = backend.readSync(deltas.dataId) as TypedArray;

  const [rtNestedSplitsData, rtDenseValuesData] = raggedRangeImplCPU(
      $starts, starts.shape, starts.dtype, $limits, limits.shape, $deltas,
      deltas.shape);

  const rtNestedSplits = backend.makeTensorInfo(
      [rtNestedSplitsData.length], 'int32', rtNestedSplitsData);
  const rtDenseValues = backend.makeTensorInfo(
      [rtDenseValuesData.length], starts.dtype, rtDenseValuesData);

  return [rtNestedSplits, rtDenseValues];
}

export const raggedRangeConfig: KernelConfig = {
  kernelName: RaggedRange,
  backendName: 'webgl',
  kernelFunc: raggedRange as unknown as KernelFunc,
};
