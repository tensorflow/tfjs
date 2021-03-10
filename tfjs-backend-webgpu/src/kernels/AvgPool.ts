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
import {AvgPool, AvgPoolAttrs, AvgPoolInputs, backend_util, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {identity} from './Identity';
import {MaxPoolWithFilterSizeEqualsOneProgram} from './maxpool_filtersizeone_webgpu';
import {Pool2DProgram} from './pool2d_webgpu';

export function avgPool(
    args: {inputs: AvgPoolInputs, backend: WebGPUBackend, attrs: AvgPoolAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {filterSize, strides, pad, dimRoundingMode} = attrs;
  const dilations = 1;
  const convInfo = backend_util.computePool2DInfo(
      x.shape as [number, number, number, number], filterSize, strides,
      dilations, pad, dimRoundingMode);
  if (convInfo.filterWidth === 1 && convInfo.filterHeight === 1 &&
      util.arraysEqual(convInfo.inShape, convInfo.outShape)) {
    return identity({inputs: {x}, backend});
  }

  let program: Pool2DProgram|MaxPoolWithFilterSizeEqualsOneProgram;
  if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1) {
    program = new MaxPoolWithFilterSizeEqualsOneProgram(convInfo);
  } else {
    program = new Pool2DProgram(convInfo, 'avg');
  }

  const dimensions = [
    convInfo.padInfo.left, convInfo.padInfo.top,      // Padding.
    convInfo.strideWidth, convInfo.strideHeight,      // Stride.
    convInfo.dilationWidth, convInfo.dilationHeight,  // Dilation.
    convInfo.inWidth, convInfo.inHeight,              // Conv dims.
    convInfo.effectiveFilterWidth,
    convInfo.effectiveFilterHeight  // Filter dims.
  ];
  const uniformData = new Int32Array(dimensions);
  return backend.runWebGPUProgram(program, [x], x.dtype, uniformData);
}

export const avgPoolConfig: KernelConfig = {
  kernelName: AvgPool,
  backendName: 'webgpu',
  kernelFunc: avgPool as {} as KernelFunc
};
