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

import {backend_util, KernelConfig, KernelFunc, MaxPoolGrad, MaxPoolGradAttrs, MaxPoolGradInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {MaxPool2DBackpropProgram} from '../max_pool_backprop_webgpu';
import {Pool2DProgram} from '../pool_webgpu';
import {assertNotComplex} from '../webgpu_util';

export function maxPoolGrad(args: {
  inputs: MaxPoolGradInputs,
  backend: WebGPUBackend,
  attrs: MaxPoolGradAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, input, output} = inputs;
  const x = input;
  assertNotComplex([input, output], 'maxPoolGrad');
  const {filterSize, strides, pad, dimRoundingMode} = attrs;

  const convInfo = backend_util.computePool2DInfo(
      x.shape as [number, number, number, number], filterSize, strides,
      1 /* dilations */, pad, dimRoundingMode);

  const maxPoolPositionsProgram = new Pool2DProgram(convInfo, 'max', true);
  let uniformData = [
    {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]},
    {type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left]},
    {type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth]},
    {type: 'int32', data: [convInfo.inHeight, convInfo.inWidth]}, {
      type: 'int32',
      data: [convInfo.effectiveFilterHeight, convInfo.effectiveFilterWidth]
    }
  ];
  const maxPoolPositions = backend.runWebGPUProgram(
      maxPoolPositionsProgram, [x], 'int32', uniformData);

  const maxPoolBackpropProgram = new MaxPool2DBackpropProgram(convInfo);
  uniformData = [
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
    {type: 'int32', data: [convInfo.outWidth]}
  ];
  const result = backend.runWebGPUProgram(
      maxPoolBackpropProgram, [dy, maxPoolPositions], x.dtype, uniformData);
  backend.disposeData(maxPoolPositions.dataId);

  return result;
}

export const maxPoolGradConfig: KernelConfig = {
  kernelName: MaxPoolGrad,
  backendName: 'webgpu',
  kernelFunc: maxPoolGrad as unknown as KernelFunc
};
