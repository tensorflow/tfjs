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

import {AddN, AddNInputs, KernelConfig, KernelFunc, TensorInfo, upcastType} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {AddNPackedProgram} from './addn_packed_webgpu';
import {identity} from './Identity';

export function addN(args: {inputs: AddNInputs, backend: WebGPUBackend}):
    TensorInfo {
  const {inputs, backend} = args;

  const tensors = inputs;
  if (tensors.length === 1) {
    return identity({inputs: {x: tensors[0]}, backend});
  }

  const dtype =
      tensors.map(t => t.dtype).reduce((d1, d2) => upcastType(d1, d2));
  const shapes = tensors.map(t => t.shape);
  const program = new AddNPackedProgram(shapes);
  return backend.runWebGPUProgram(program, tensors, dtype);
}

export const addNConfig: KernelConfig = {
  kernelName: AddN,
  backendName: 'webgpu',
  kernelFunc: addN as {} as KernelFunc
};
