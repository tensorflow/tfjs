/**
 * @license
 * Copyright 2023 Google LLC.
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

import {backend_util, Conv3D, Conv3DAttrs, Conv3DInputs, KernelConfig, KernelFunc, upcastType} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {Conv3DNaiveProgram} from '../conv3d_naive_webgpu';

export function conv3D(
    args: {inputs: Conv3DInputs, attrs: Conv3DAttrs, backend: WebGPUBackend}) {
  const {inputs, backend, attrs} = args;
  const {x, filter} = inputs;
  const {strides, pad, dilations} = attrs;

  const convInfo = backend_util.computeConv3DInfo(
      x.shape as [number, number, number, number, number],
      filter.shape as [number, number, number, number, number], strides,
      dilations, pad);

  const padInfo =
      [convInfo.padInfo.front, convInfo.padInfo.top, convInfo.padInfo.left];
  const dimensions = [
    {
      type: 'int32',
      data: [convInfo.filterDepth, convInfo.filterHeight, convInfo.filterWidth]
    },
    {type: 'int32', data: [...padInfo]}, {
      type: 'int32',
      data: [convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth]
    },
    {
      type: 'int32',
      data: [
        convInfo.dilationDepth, convInfo.dilationHeight, convInfo.dilationWidth
      ]
    }
  ];
  const program = new Conv3DNaiveProgram(convInfo);
  const dtype = upcastType(x.dtype, filter.dtype);
  return backend.runWebGPUProgram(program, [x, filter], dtype, dimensions);
}

export const conv3DConfig: KernelConfig = {
  kernelName: Conv3D,
  backendName: 'webgpu',
  kernelFunc: conv3D as {} as KernelFunc,
};
