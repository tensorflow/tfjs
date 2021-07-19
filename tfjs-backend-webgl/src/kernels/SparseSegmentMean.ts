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

import {KernelConfig, KernelFunc, SparseSegmentMean, SparseSegmentMeanInputs, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {sparseSegmentReductionImplCPU} from '../kernel_utils/shared';

export function sparseSegmentMean(
    args: {inputs: SparseSegmentMeanInputs, backend: MathBackendWebGL}):
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

  const $data = backend.readSync(data.dataId) as TypedArray;
  const $indices = backend.readSync(indices.dataId) as TypedArray;
  const $segmentIds = backend.readSync(segmentIds.dataId) as TypedArray;

  const [outputData, outputDataShape] = sparseSegmentReductionImplCPU(
      $data, data.shape, data.dtype, $indices, $segmentIds, true);
  return backend.makeTensorInfo(outputDataShape, data.dtype, outputData);
}

export const sparseSegmentMeanConfig: KernelConfig = {
  kernelName: SparseSegmentMean,
  backendName: 'webgl',
  kernelFunc: sparseSegmentMean as {} as KernelFunc,
};
