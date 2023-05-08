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

import {Any, AnyAttrs, AnyInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {reduce} from '../kernel_utils/reduce';

export function any(
    args: {inputs: AnyInputs, attrs: AnyAttrs, backend: WebGPUBackend}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {keepDims, axis} = attrs;

  return reduce(x, axis, keepDims, 'any', backend);
}

export const anyConfig: KernelConfig = {
  kernelName: Any,
  backendName: 'webgpu',
  kernelFunc: any as unknown as KernelFunc
};
