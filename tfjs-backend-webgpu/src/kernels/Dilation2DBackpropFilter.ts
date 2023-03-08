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

import {backend_util, Dilation2DAttrs, Dilation2DBackpropFilter, Dilation2DBackpropFilterInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {Dilation2DBackpropFilterProgram} from '../dilation_backprop_webgpu';
import {fill} from './Fill';

export function dilation2DBackpropFilter(args: {
  inputs: Dilation2DBackpropFilterInputs,
  attrs: Dilation2DAttrs,
  backend: WebGPUBackend
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, filter, dy} = inputs;
  const {strides, pad, dilations} = attrs;

  const convInfo = backend_util.computeDilation2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number], strides, pad,
      'NHWC' /* dataFormat */, dilations);

  const dtype = filter.dtype;
  const program =
      new Dilation2DBackpropFilterProgram(convInfo, filter.shape, dtype);
  const uniformData = [
    {type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth]},
    {type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left]},
    {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]},
    {type: 'int32', data: [convInfo.dilationHeight, convInfo.dilationWidth]},
    {type: 'int32', data: [util.sizeFromShape(convInfo.outShape)]}
  ];
  const output = fill({backend, attrs: {shape: filter.shape, value: 0, dtype}});
  return backend.runWebGPUProgram(
      program, [x, filter, dy], dtype, uniformData, output);
}

export const dilation2DBackpropFilterConfig: KernelConfig = {
  kernelName: Dilation2DBackpropFilter,
  backendName: 'webgpu',
  kernelFunc: dilation2DBackpropFilter as unknown as KernelFunc
};
