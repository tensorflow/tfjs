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

import {KernelConfig, KernelFunc, Softmax, SoftmaxAttrs, SoftmaxInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {SoftmaxProgram} from '../softmax_webgpu';

import {reshape} from './Reshape';

export function softmax(
    args: {inputs: SoftmaxInputs, backend: WebGPUBackend, attrs: SoftmaxAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {logits} = inputs;
  const {dim} = attrs;

  const logitsReshaped = reshape({
    inputs: {x: logits},
    backend,
    attrs: {
      shape: [
        util.sizeFromShape(logits.shape) / logits.shape[dim], logits.shape[dim]
      ]
    }
  });
  const program = new SoftmaxProgram(logitsReshaped.shape);
  const res = backend.runWebGPUProgram(program, [logitsReshaped], logits.dtype);
  const resReshaped =
      reshape({inputs: {x: res}, backend, attrs: {shape: logits.shape}});
  backend.disposeData(logitsReshaped.dataId);
  backend.disposeData(res.dataId);
  return resReshaped;
}

export const softmaxConfig: KernelConfig = {
  kernelName: Softmax,
  backendName: 'webgpu',
  kernelFunc: softmax as unknown as KernelFunc
};
