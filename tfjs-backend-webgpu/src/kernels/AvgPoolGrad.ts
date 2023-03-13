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

import {AvgPoolGrad, AvgPoolGradAttrs, AvgPoolGradInputs, backend_util, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {AvgPool2DBackpropProgram} from '../avg_pool_backprop_webgpu';
import {WebGPUBackend} from '../backend_webgpu';
import {assertNotComplex} from '../webgpu_util';

export function avgPoolGrad(args: {
  inputs: AvgPoolGradInputs,
  backend: WebGPUBackend,
  attrs: AvgPoolGradAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, input} = inputs;
  const x = input;
  assertNotComplex([dy, input], 'avgPoolGrad');
  const {filterSize, strides, pad} = attrs;

  const convInfo = backend_util.computePool2DInfo(
      x.shape as [number, number, number, number], filterSize, strides,
      1 /* dilations */, pad);
  const program = new AvgPool2DBackpropProgram(convInfo);
  const avgMultiplier = 1 / (convInfo.filterHeight * convInfo.filterWidth);
  const uniformData = [
    {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]}, {
      type: 'int32',
      data: [
        convInfo.effectiveFilterHeight - 1 - convInfo.padInfo.top,
        convInfo.effectiveFilterWidth - 1 - convInfo.padInfo.left
      ]
    },
    {type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth]}, {
      type: 'int32',
      data: [convInfo.effectiveFilterHeight, convInfo.effectiveFilterWidth]
    },
    {type: 'int32', data: [convInfo.outHeight]},
    {type: 'int32', data: [convInfo.outWidth]},
    {type: 'float32', data: [avgMultiplier]}
  ];
  return backend.runWebGPUProgram(program, [dy], x.dtype, uniformData);
}

export const avgPoolGradConfig: KernelConfig = {
  kernelName: AvgPoolGrad,
  backendName: 'webgpu',
  kernelFunc: avgPoolGrad as unknown as KernelFunc
};
