/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {KernelConfig, KernelFunc, TensorInfo, ZerosLike, ZerosLikeInputs} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {fill} from './Fill';

export function zerosLike(
    args: {inputs: ZerosLikeInputs, backend: WebGPUBackend}): TensorInfo {
  const {inputs, backend} = args;
  const {x} = inputs;

  return fill({
    attrs:
        {shape: x.shape, dtype: x.dtype, value: x.dtype === 'string' ? '' : 0},
    backend
  });
}

export const zerosLikeConfig: KernelConfig = {
  kernelName: ZerosLike,
  backendName: 'webgpu',
  kernelFunc: zerosLike as {} as KernelFunc
};
