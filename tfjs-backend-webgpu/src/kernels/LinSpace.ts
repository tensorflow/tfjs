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

import {KernelConfig, KernelFunc, LinSpace, LinSpaceAttrs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {LinSpaceProgram} from '../lin_space_webgpu';

export function linSpace(args: {backend: WebGPUBackend, attrs: LinSpaceAttrs}):
    TensorInfo {
  const {backend, attrs} = args;
  const {start, stop, num} = attrs;
  const step = (stop - start) / (num - 1);

  const program = new LinSpaceProgram(num);
  const uniformData =
      [{type: 'float32', data: [start]}, {type: 'float32', data: [step]}];
  return backend.runWebGPUProgram(program, [], 'float32', uniformData);
}

export const linSpaceConfig: KernelConfig = {
  kernelName: LinSpace,
  backendName: 'webgpu',
  kernelFunc: linSpace as unknown as KernelFunc
};
