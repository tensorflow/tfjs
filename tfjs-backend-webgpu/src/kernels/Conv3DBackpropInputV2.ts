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

import {backend_util, Conv3DBackpropInputV2, Conv3DBackpropInputV2Attrs, Conv3DBackpropInputV2Inputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {Conv3DDerInputProgram} from '../conv_backprop_webgpu';

export function conv3DBackpropInputV2(args: {
  inputs: Conv3DBackpropInputV2Inputs,
  attrs: Conv3DBackpropInputV2Attrs,
  backend: WebGPUBackend
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, filter} = inputs;
  const {strides, pad, inputShape} = attrs;

  const convInfo = backend_util.computeConv3DInfo(
      inputShape, filter.shape as [number, number, number, number, number],
      strides, 1 /* dilations */, pad);

  const program = new Conv3DDerInputProgram(convInfo);
  const uniformData = [
    {
      type: 'int32',
      data: [convInfo.filterDepth, convInfo.filterHeight, convInfo.filterWidth]
    },
    {
      type: 'int32',
      data: [
        convInfo.filterDepth - 1 - convInfo.padInfo.front,
        convInfo.filterHeight - 1 - convInfo.padInfo.top,
        convInfo.filterWidth - 1 - convInfo.padInfo.left
      ]
    },
    {
      type: 'int32',
      data: [convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth]
    },
    {type: 'int32', data: [convInfo.outDepth]},
    {type: 'int32', data: [convInfo.outHeight]},
    {type: 'int32', data: [convInfo.outWidth]},
    {type: 'int32', data: [convInfo.outChannels]}
  ];

  return backend.runWebGPUProgram(program, [dy, filter], dy.dtype, uniformData);
}

export const conv3DBackpropInputV2Config: KernelConfig = {
  kernelName: Conv3DBackpropInputV2,
  backendName: 'webgpu',
  kernelFunc: conv3DBackpropInputV2 as unknown as KernelFunc,
};
