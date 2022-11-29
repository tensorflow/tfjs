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

import {KernelConfig, KernelFunc, Reverse, ReverseAttrs, ReverseInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {ReverseProgram} from '../reverse_webgpu';

import {identity} from './Identity';
import {reshape} from './Reshape';

export function reverse(
    args: {inputs: ReverseInputs, backend: WebGPUBackend, attrs: ReverseAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {dims} = attrs;

  const xRank = x.shape.length;
  if (xRank === 0) {
    return identity({inputs: {x}, backend});
  }

  const xShape = x.shape;
  const xShape4D: [number, number, number, number] = [1, 1, 1, 1];
  xShape.forEach((d, i) => {
    const index = i + 4 - xRank;
    xShape4D[index] = d;
  });

  const axes = util.parseAxisParam(dims, x.shape);
  const dims4D: [number, number, number, number] = [0, 0, 0, 0];
  axes.forEach(ax => {
    const index = ax + 4 - xRank;
    dims4D[index] = 1;
  });
  const uniformData = [{type: 'int32', data: dims4D}];

  const xReshaped = reshape({inputs: {x}, backend, attrs: {shape: xShape4D}});

  const program = new ReverseProgram(xShape4D);
  const values = backend.runWebGPUProgram(
      program, [xReshaped], xReshaped.dtype, uniformData);
  backend.disposeData(xReshaped.dataId);

  const result =
      reshape({inputs: {x: values}, backend, attrs: {shape: xShape}});
  backend.disposeData(values.dataId);

  return result;
}

export const reverseConfig: KernelConfig = {
  kernelName: Reverse,
  backendName: 'webgpu',
  kernelFunc: reverse as unknown as KernelFunc
};
