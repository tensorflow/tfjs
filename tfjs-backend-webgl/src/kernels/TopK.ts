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

import {KernelConfig, KernelFunc, NumericDataType, TensorInfo, TopK, TopKAttrs, TopKInputs, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {topKImplCPU} from '../kernel_utils/shared';

export function topK(
    args: {inputs: TopKInputs, backend: MathBackendWebGL, attrs: TopKAttrs}):
    TensorInfo[] {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {k, sorted} = attrs;

  const xVals = backend.readSync(x.dataId) as TypedArray;
  const [allTopKVals, allTopKIndices] =
      topKImplCPU(xVals, x.shape, x.dtype as NumericDataType, k, sorted);

  return [
    backend.makeTensorInfo(
        allTopKVals.shape, allTopKVals.dtype, allTopKVals.values),
    backend.makeTensorInfo(
        allTopKIndices.shape, allTopKIndices.dtype, allTopKIndices.values)
  ];
}

export const topKConfig: KernelConfig = {
  kernelName: TopK,
  backendName: 'webgl',
  kernelFunc: topK as {} as KernelFunc
};
