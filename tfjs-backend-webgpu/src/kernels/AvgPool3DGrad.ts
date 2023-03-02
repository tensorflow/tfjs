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

import {AvgPool3DGrad, AvgPool3DGradAttrs, AvgPool3DGradInputs, backend_util, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {AvgPool3DBackpropProgram} from '../avg_pool_backprop_webgpu';
import {WebGPUBackend} from '../backend_webgpu';

export function avgPool3DGrad(args: {
  inputs: AvgPool3DGradInputs,
  backend: WebGPUBackend,
  attrs: AvgPool3DGradAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, input} = inputs;
  const x = input;
  const {filterSize, strides, pad, dimRoundingMode} = attrs;

  const convInfo = backend_util.computePool3DInfo(
      x.shape as [number, number, number, number, number], filterSize, strides,
      1 /* dilations */, pad, dimRoundingMode);
  const program = new AvgPool3DBackpropProgram(convInfo);
  const avgMultiplier =
      1 / (convInfo.filterDepth * convInfo.filterHeight * convInfo.filterWidth);
  const uniformData = [
    {
      type: 'int32',
      data: [convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth]
    },
    {
      type: 'int32',
      data: [
        convInfo.effectiveFilterDepth - 1 - convInfo.padInfo.front,
        convInfo.effectiveFilterHeight - 1 - convInfo.padInfo.top,
        convInfo.effectiveFilterWidth - 1 - convInfo.padInfo.left
      ]
    },
    {
      type: 'int32',
      data: [
        convInfo.effectiveFilterDepth, convInfo.effectiveFilterHeight,
        convInfo.effectiveFilterWidth
      ]
    },
    {type: 'int32', data: [convInfo.outDepth]},
    {type: 'int32', data: [convInfo.outHeight]},
    {type: 'int32', data: [convInfo.outWidth]},
    {type: 'float32', data: [avgMultiplier]}
  ];
  return backend.runWebGPUProgram(program, [dy], x.dtype, uniformData);
}

export const avgPool3DGradConfig: KernelConfig = {
  kernelName: AvgPool3DGrad,
  backendName: 'webgpu',
  kernelFunc: avgPool3DGrad as unknown as KernelFunc
};
