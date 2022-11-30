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

import {KernelConfig, KernelFunc, OneHot, OneHotAttrs, OneHotInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {OneHotProgram} from '../onehot_webgpu';
import {reshape} from './Reshape';

export function oneHot(
    args: {inputs: OneHotInputs, backend: WebGPUBackend, attrs: OneHotAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {indices} = inputs;
  const {dtype, depth, onValue, offValue} = attrs;

  const indicesSize = util.sizeFromShape(indices.shape);
  const program = new OneHotProgram(indicesSize, depth);
  const reshaped =
      reshape({inputs: {x: indices}, backend, attrs: {shape: [indicesSize]}});

  const uniformData =
      [{type: 'float32', data: [onValue]}, {type: 'float32', data: [offValue]}];
  const result =
      backend.runWebGPUProgram(program, [reshaped], dtype, uniformData);
  backend.disposeData(reshaped.dataId);

  const outShape = [...indices.shape, depth];
  const out = reshape({inputs: {x: result}, backend, attrs: {shape: outShape}});
  backend.disposeData(result.dataId);

  return out;
}

export const oneHotConfig: KernelConfig = {
  kernelName: OneHot,
  backendName: 'webgpu',
  kernelFunc: oneHot as unknown as KernelFunc
};
