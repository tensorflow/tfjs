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

import {KernelConfig, KernelFunc, LRN, LRNAttrs, LRNInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {LRNProgram, LRNSharedProgram} from '../lrn_webgpu';

export function lrn(
    args: {inputs: LRNInputs, backend: WebGPUBackend, attrs: LRNAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {depthRadius, bias, alpha, beta} = attrs;

  // When the adjacent channels is less than or equal to 16, which could cover
  // most cases, we use shared memory version to get better performance.
  // The theoretical adjacent channels may be very large, but the shared memory
  // size of hardware is limited, so we use the naive version when the adjacent
  // channels is large.
  let program: LRNProgram|LRNSharedProgram;
  if (depthRadius > 16) {
    program = new LRNProgram(x.shape);
  } else {
    program = new LRNSharedProgram(x.shape, depthRadius);
  }
  const uniformData = [
    {type: 'int32', data: [depthRadius]}, {type: 'float32', data: [bias]},
    {type: 'float32', data: [alpha]}, {type: 'float32', data: [beta]}
  ];
  const res = backend.runWebGPUProgram(program, [x], x.dtype, uniformData);

  return res;
}

export const lrnConfig: KernelConfig = {
  kernelName: LRN,
  backendName: 'webgpu',
  kernelFunc: lrn as unknown as KernelFunc
};
