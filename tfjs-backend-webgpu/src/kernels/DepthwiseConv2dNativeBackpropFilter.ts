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

import {backend_util, DepthwiseConv2dNativeBackpropFilter, DepthwiseConv2dNativeBackpropFilterAttrs, DepthwiseConv2dNativeBackpropFilterInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {DepthwiseConv2DDerFilterProgram} from '../conv_backprop_depthwise_webgpu';

export function depthwiseConv2dNativeBackpropFilter(args: {
  inputs: DepthwiseConv2dNativeBackpropFilterInputs,
  attrs: DepthwiseConv2dNativeBackpropFilterAttrs,
  backend: WebGPUBackend
}) {
  const {inputs, backend, attrs} = args;
  const {x, dy} = inputs;
  const {strides, dilations, pad, dimRoundingMode, filterShape} = attrs;

  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number], filterShape, strides,
      dilations, pad, dimRoundingMode, true /* depthwise */);

  const program = new DepthwiseConv2DDerFilterProgram(convInfo);
  const uniformData = [
    {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]},
    {type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left]},
    {type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth]},
    {type: 'int32', data: [convInfo.outHeight]},
    {type: 'int32', data: [convInfo.outWidth]},
    {type: 'int32', data: [convInfo.inHeight]},
    {type: 'int32', data: [convInfo.inWidth]},
    {type: 'int32', data: [convInfo.batchSize]},
    {type: 'int32', data: [convInfo.outChannels / convInfo.inChannels]}
  ];
  return backend.runWebGPUProgram(program, [x, dy], 'float32', uniformData);
}

export const depthwiseConv2dNativeBackpropFilterConfig: KernelConfig = {
  kernelName: DepthwiseConv2dNativeBackpropFilter,
  backendName: 'webgpu',
  kernelFunc: depthwiseConv2dNativeBackpropFilter as unknown as KernelFunc
};
