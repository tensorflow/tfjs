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

import {backend_util, Dilation2D, Dilation2DAttrs, Dilation2DInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {Dilation2DProgram} from '../dilation_webgpu';

export function dilation2D(args: {
  inputs: Dilation2DInputs,
  attrs: Dilation2DAttrs,
  backend: WebGPUBackend
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, filter} = inputs;
  const {strides, pad, dilations} = attrs;

  const convInfo = backend_util.computeDilation2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number], strides, pad,
      'NHWC' /* dataFormat */, dilations);
  const padInfo = [convInfo.padInfo.top, convInfo.padInfo.left];
  const uniformData = [
    {type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth]},
    {type: 'int32', data: [...padInfo]},
    {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]},
    {type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth]}
  ];

  const program = new Dilation2DProgram(convInfo);
  const out =
      backend.runWebGPUProgram(program, [x, filter], x.dtype, uniformData);

  return out;
}

export const dilation2DConfig: KernelConfig = {
  kernelName: Dilation2D,
  backendName: 'webgpu',
  kernelFunc: dilation2D as unknown as KernelFunc
};
