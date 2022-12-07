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

import {KernelConfig, KernelFunc, SearchSorted, SearchSortedAttrs, SearchSortedInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {SearchSortedProgram} from '../search_sorted_webgpu';

export function searchSorted(args: {
  inputs: SearchSortedInputs,
  backend: WebGPUBackend,
  attrs: SearchSortedAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {sortedSequence, values} = inputs;
  const {side} = attrs;

  const program =
      new SearchSortedProgram([values.shape[0], values.shape[1]], side);
  const uniformData = [{type: 'int32', data: [sortedSequence.shape[1]]}];
  return backend.runWebGPUProgram(
      program, [sortedSequence, values], 'int32', uniformData);
}

export const searchSortedConfig: KernelConfig = {
  kernelName: SearchSorted,
  backendName: 'webgpu',
  kernelFunc: searchSorted as unknown as KernelFunc,
};
