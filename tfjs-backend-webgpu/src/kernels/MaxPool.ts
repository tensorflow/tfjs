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
import {backend_util, KernelConfig, KernelFunc, MaxPool, MaxPoolAttrs, MaxPoolInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {identity} from './Identity';
import {Pool2DProgram} from '../pool2d_webgpu';
import {PoolWithFilterSizeEqualsOneProgram} from '../pool_filtersizeone_webgpu';

export function maxPool(
    args: {inputs: MaxPoolInputs, backend: WebGPUBackend, attrs: MaxPoolAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {filterSize, strides, pad, dimRoundingMode} = attrs;
  const dilations = 1;
  const convInfo = backend_util.computePool2DInfo(
      x.shape as [number, number, number, number], filterSize, strides,
      dilations, pad, dimRoundingMode);
  let program: Pool2DProgram|PoolWithFilterSizeEqualsOneProgram;
  const dimensions = [];
  if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1) {
    if (util.arraysEqual(convInfo.inShape, convInfo.outShape)) {
      return identity({inputs: {x}, backend});
    }
    program = new PoolWithFilterSizeEqualsOneProgram(convInfo);
    dimensions.push(
        {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]});
  } else {
    program = new Pool2DProgram(convInfo, 'max');
    dimensions.push(
        {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]},
        {type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left]}, {
          type: 'int32',
          data: [convInfo.dilationHeight, convInfo.dilationWidth]
        },
        {type: 'int32', data: [convInfo.inHeight, convInfo.inWidth]}, {
          type: 'int32',
          data: [convInfo.effectiveFilterHeight, convInfo.effectiveFilterWidth]
        });
  }

  return backend.runWebGPUProgram(program, [x], x.dtype, dimensions);
}

export const maxPoolConfig: KernelConfig = {
  kernelName: MaxPool,
  backendName: 'webgpu',
  kernelFunc: maxPool as {} as KernelFunc
};
