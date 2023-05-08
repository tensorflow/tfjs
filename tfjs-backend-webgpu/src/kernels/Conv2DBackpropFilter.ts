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

import {backend_util, Conv2DBackpropFilter, Conv2DBackpropFilterAttrs, Conv2DBackpropFilterInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {Conv2DDerFilterProgram} from '../conv_backprop_webgpu';

export function conv2DBackpropFilter(args: {
  inputs: Conv2DBackpropFilterInputs,
  backend: WebGPUBackend,
  attrs: Conv2DBackpropFilterAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, dy} = inputs;
  const {strides, pad, dataFormat, dimRoundingMode, filterShape} = attrs;

  const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number], filterShape, strides,
      1 /* dilations */, pad, dimRoundingMode, false /* depthwise */,
      $dataFormat);

  const program = new Conv2DDerFilterProgram(convInfo);
  const uniformData = [
    {type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left]},
    {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]},
    {type: 'int32', data: [convInfo.batchSize]},
    {type: 'int32', data: [convInfo.outHeight]},
    {type: 'int32', data: [convInfo.outWidth]},
    {type: 'int32', data: [convInfo.inHeight]},
    {type: 'int32', data: [convInfo.inWidth]}
  ];
  return backend.runWebGPUProgram(program, [x, dy], x.dtype, uniformData);
}

export const conv2DBackpropFilterConfig: KernelConfig = {
  kernelName: Conv2DBackpropFilter,
  backendName: 'webgpu',
  kernelFunc: conv2DBackpropFilter as unknown as KernelFunc
};
