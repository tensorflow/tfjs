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

import {KernelConfig, KernelFunc, LRNGrad, LRNGradAttrs, LRNGradInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {LRNGradProgram} from '../lrn_grad_webgpu';

export function lrnGrad(
    args: {inputs: LRNGradInputs, backend: WebGPUBackend, attrs: LRNGradAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, y, dy} = inputs;
  const {depthRadius, bias, alpha, beta} = attrs;

  const program = new LRNGradProgram(x.shape);
  const uniformData = [
    {type: 'int32', data: [depthRadius]}, {type: 'float32', data: [bias]},
    {type: 'float32', data: [alpha]}, {type: 'float32', data: [beta]}
  ];
  const res =
      backend.runWebGPUProgram(program, [x, y, dy], x.dtype, uniformData);

  return res;
}

export const lrnGradConfig: KernelConfig = {
  kernelName: LRNGrad,
  backendName: 'webgpu',
  kernelFunc: lrnGrad as unknown as KernelFunc
};
