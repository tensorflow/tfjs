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

import {Cumprod, CumprodAttrs, CumprodInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {CumOpType} from '../cum_webgpu';
import {cumImpl} from './Cum_impl';

export function cumprod(
    args: {inputs: CumprodInputs, backend: WebGPUBackend, attrs: CumprodAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {axis, exclusive, reverse} = attrs;
  return cumImpl(CumOpType.Prod, x, backend, axis, exclusive, reverse);
}

export const cumprodConfig: KernelConfig = {
  kernelName: Cumprod,
  backendName: 'webgpu',
  kernelFunc: cumprod as {} as KernelFunc
};
