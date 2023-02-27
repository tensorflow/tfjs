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
import {AvgPool3D, AvgPool3DAttrs, AvgPool3DInputs, backend_util, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {Pool3DProgram} from '../pool_webgpu';

export function avgPool3D(args: {
  inputs: AvgPool3DInputs,
  backend: WebGPUBackend,
  attrs: AvgPool3DAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {filterSize, strides, pad, dataFormat, dimRoundingMode} = attrs;
  const dilations: [number, number, number] = [1, 1, 1];

  const convInfo = backend_util.computePool3DInfo(
      x.shape as [number, number, number, number, number], filterSize, strides,
      dilations, pad, dimRoundingMode, dataFormat);
  const avgPoolProgram = new Pool3DProgram(convInfo, 'avg');
  const dimensions = [
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
  return backend.runWebGPUProgram(avgPoolProgram, [x], x.dtype, dimensions);
}

export const avgPool3DConfig: KernelConfig = {
  kernelName: AvgPool3D,
  backendName: 'webgpu',
  kernelFunc: avgPool3D as unknown as KernelFunc
};
