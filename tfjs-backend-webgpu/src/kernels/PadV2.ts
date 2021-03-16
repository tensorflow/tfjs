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

import {KernelConfig, KernelFunc, PadV2, PadV2Attrs, PadV2Inputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {PadProgram} from './pad_webgpu';

export const padV2 =
    (args: {inputs: PadV2Inputs, backend: WebGPUBackend, attrs: PadV2Attrs}):
        TensorInfo => {
          const {inputs, backend, attrs} = args;
          const {x} = inputs;
          const {paddings, constantValue} = attrs;
          const uniformData = new Float32Array([constantValue]);
          const program = new PadProgram(x.shape, paddings);
          return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
        };

export const padV2Config: KernelConfig = {
  kernelName: PadV2,
  backendName: 'webgpu',
  kernelFunc: padV2 as {} as KernelFunc
};
