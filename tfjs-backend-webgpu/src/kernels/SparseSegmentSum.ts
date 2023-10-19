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

import {KernelConfig, KernelFunc, SparseSegmentSum, SparseSegmentSumInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {sparseSegmentReduce} from '../kernel_utils/sparse_segment_reduce';

export function sparseSegmentSum(
    args: {inputs: SparseSegmentSumInputs, backend: WebGPUBackend}):
    TensorInfo {
  const {inputs, backend} = args;
  const {data, indices, segmentIds} = inputs;

  return sparseSegmentReduce(data, indices, segmentIds, true, backend);
}

export const sparseSegmentSumConfig: KernelConfig = {
  kernelName: SparseSegmentSum,
  backendName: 'webgpu',
  kernelFunc: sparseSegmentSum as unknown as KernelFunc,
};
