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

import {Diag, DiagInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {DiagProgram} from '../diag_webgpu';
import {reshape} from './Reshape';

export function diag(args: {inputs: DiagInputs, backend: WebGPUBackend}):
    TensorInfo {
  const {inputs, backend} = args;
  const {x} = inputs;

  const outShape = [...x.shape, ...x.shape];
  const xSize = util.sizeFromShape(x.shape);

  const flat = reshape({inputs: {x}, backend, attrs: {shape: [xSize]}});

  const program = new DiagProgram(xSize);
  const res = backend.runWebGPUProgram(program, [flat], flat.dtype);

  const out = reshape({inputs: {x: res}, backend, attrs: {shape: outShape}});

  backend.disposeData(flat.dataId);
  backend.disposeData(res.dataId);

  return out;
}

export const diagConfig: KernelConfig = {
  kernelName: Diag,
  backendName: 'webgpu',
  kernelFunc: diag as unknown as KernelFunc
};
