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

import {KernelConfig, KernelFunc, SearchSorted, SearchSortedAttrs, SearchSortedInputs, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

import {searchSortedImpl} from './SearchSorted_impl';

export function searchSorted(args: {
  inputs: SearchSortedInputs,
  backend: MathBackendCPU,
  attrs: SearchSortedAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {sortedSequence, values} = inputs;
  const {side} = attrs;

  const $sortedSequence =
      backend.data.get(sortedSequence.dataId).values as TypedArray;
  const $values = backend.data.get(values.dataId).values as TypedArray;

  const output = searchSortedImpl(
      $sortedSequence, $values, sortedSequence.shape[0],
      sortedSequence.shape[1], values.shape[1], side);
  return backend.makeTensorInfo(values.shape, 'int32', output);
}

export const searchSortedConfig: KernelConfig = {
  kernelName: SearchSorted,
  backendName: 'cpu',
  kernelFunc: searchSorted as {} as KernelFunc,
};
