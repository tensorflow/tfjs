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

import {engine, KernelConfig, KernelFunc, Reshape, ReshapeAttrs, ReshapeInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

export function reshape(args: {
  inputs: ReshapeInputs,
  backend: WebGPUBackend,
  attrs: ReshapeAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {shape} = attrs;
  const webgpuBackend = backend;

  return engine().makeTensorFromDataId(x.dataId, shape, x.dtype, webgpuBackend) as
        TensorInfo;
}

export const reshapeConfig: KernelConfig = {
  kernelName: Reshape,
  backendName: 'webgpu',
  kernelFunc: reshape as {} as KernelFunc
};
