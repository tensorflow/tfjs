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

import {KernelConfig, SparseSegmentMean, SparseSegmentMeanInputs, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

import {sparseSegmentReductionImpl} from './SparseSegmentReduction_impl';

export function sparseSegmentMean(
    args: {inputs: SparseSegmentMeanInputs, backend: MathBackendCPU}):
    TensorInfo {
  const {inputs, backend} = args;
  const {data, indices, segmentIds} = inputs;
  if (data.shape.length < 1) {
    throw new Error(
        `Data should be at least 1 dimensional but received scalar`);
  }
  if (indices.shape.length !== 1) {
    throw new Error(`Indices should be a vector but received shape
          ${indices.shape}`);
  }
  if (segmentIds.shape.length !== 1) {
    throw new Error(`Segment ids should be a vector but received shape
          ${segmentIds.shape}`);
  }

  const $data = backend.data.get(data.dataId).values as TypedArray;
  const $indices = backend.data.get(indices.dataId).values as TypedArray;
  const $segmentIds = backend.data.get(segmentIds.dataId).values as TypedArray;

  const [outputData, outputDataShape] = sparseSegmentReductionImpl(
      $data, data.shape, data.dtype, $indices, $segmentIds, true);
  return backend.makeTensorInfo(outputDataShape, data.dtype, outputData);
}

export const sparseSegmentMeanConfig: KernelConfig = {
  kernelName: SparseSegmentMean,
  backendName: 'cpu',
  kernelFunc: sparseSegmentMean,
};
