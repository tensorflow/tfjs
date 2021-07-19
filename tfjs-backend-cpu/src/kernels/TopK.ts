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

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {topKImpl} from './TopK_impl';

export function topK(
    args: {inputs: TopKInputs, backend: MathBackendCPU, attrs: TopKAttrs}):
    [TensorInfo, TensorInfo] {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {k, sorted} = attrs;

  assertNotComplex(x, 'topk');

  const xVals = backend.data.get(x.dataId).values as TypedArray;
  const [allTopKVals, allTopKIndices] =
      topKImpl(xVals, x.shape, x.dtype as NumericDataType, k, sorted);

  return [
    backend.makeTensorInfo(
        allTopKVals.shape, allTopKVals.dtype, allTopKVals.values),
    backend.makeTensorInfo(
        allTopKIndices.shape, allTopKIndices.dtype, allTopKIndices.values)
  ];
}

export const topKConfig: KernelConfig = {
  kernelName: TopK,
  backendName: 'cpu',
  kernelFunc: topK as {} as KernelFunc
};
