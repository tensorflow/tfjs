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

import {backend_util, KernelConfig, KernelFunc, MaxPool3DGrad, MaxPool3DGradAttrs, MaxPool3DGradInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {MaxPool3DBackpropProgram} from '../max_pool_backprop_webgpu';
import {Pool3DProgram} from '../pool_webgpu';

export function maxPool3DGrad(args: {
  inputs: MaxPool3DGradInputs,
  backend: WebGPUBackend,
  attrs: MaxPool3DGradAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, input} = inputs;
  const x = input;
  const {filterSize, strides, pad, dimRoundingMode} = attrs;
  const dilations: [number, number, number] = [1, 1, 1];

  const convInfo = backend_util.computePool3DInfo(
      x.shape as [number, number, number, number, number], filterSize, strides,
      dilations, pad, dimRoundingMode);

  const maxPool3dPositionsProgram =
      new Pool3DProgram(convInfo, 'max', true /* get positions */);
  let uniformData = [
    {
      type: 'int32',
      data: [convInfo.strideDepth, convInfo.strideHeight, convInfo.strideWidth]
    },
    {
      type: 'int32',
      data:
          [convInfo.padInfo.front, convInfo.padInfo.top, convInfo.padInfo.left]
    },
    {
      type: 'int32',
      data: [convInfo.inDepth, convInfo.inHeight, convInfo.inWidth]
    },
    {
      type: 'int32',
      data: [
        convInfo.effectiveFilterDepth, convInfo.effectiveFilterHeight,
        convInfo.effectiveFilterWidth
      ]
    }
  ];
  const maxPool3dPositions = backend.runWebGPUProgram(
      maxPool3dPositionsProgram, [x], 'int32', uniformData);

  const maxPool3dBackpropProgram = new MaxPool3DBackpropProgram(convInfo);
  uniformData = [
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
    {type: 'int32', data: [convInfo.outWidth]}
  ];
  const result = backend.runWebGPUProgram(
      maxPool3dBackpropProgram, [dy, maxPool3dPositions], x.dtype, uniformData);
  backend.disposeData(maxPool3dPositions.dataId);

  return result;
}

export const maxPool3DGradConfig: KernelConfig = {
  kernelName: MaxPool3DGrad,
  backendName: 'webgpu',
  kernelFunc: maxPool3DGrad as unknown as KernelFunc
};
