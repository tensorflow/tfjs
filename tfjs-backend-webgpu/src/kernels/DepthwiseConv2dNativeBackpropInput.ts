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

import {backend_util, DepthwiseConv2dNativeBackpropInput, DepthwiseConv2dNativeBackpropInputAttrs, DepthwiseConv2dNativeBackpropInputInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {DepthwiseConv2DDerInputProgram} from '../conv_backprop_depthwise_webgpu';

export function depthwiseConv2dNativeBackpropInput(args: {
  inputs: DepthwiseConv2dNativeBackpropInputInputs,
  attrs: DepthwiseConv2dNativeBackpropInputAttrs,
  backend: WebGPUBackend
}) {
  const {inputs, backend, attrs} = args;
  const {dy, filter} = inputs;
  const {strides, dilations, pad, dimRoundingMode, inputShape} = attrs;

  const convInfo = backend_util.computeConv2DInfo(
      inputShape, filter.shape as [number, number, number, number], strides,
      dilations, pad, dimRoundingMode, true /* depthwise */);

  const program = new DepthwiseConv2DDerInputProgram(convInfo);
  const uniformData = [
    {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]}, {
      type: 'int32',
      data: [
        convInfo.filterHeight - 1 - convInfo.padInfo.top,
        convInfo.filterWidth - 1 - convInfo.padInfo.left
      ]
    },
    {type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth]},
    {type: 'int32', data: [convInfo.outHeight]},
    {type: 'int32', data: [convInfo.outWidth]},
    {type: 'int32', data: [convInfo.outChannels / convInfo.inChannels]}
  ];
  return backend.runWebGPUProgram(program, [dy, filter], dy.dtype, uniformData);
}

export const depthwiseConv2dNativeBackpropInputConfig: KernelConfig = {
  kernelName: DepthwiseConv2dNativeBackpropInput,
  backendName: 'webgpu',
  kernelFunc: depthwiseConv2dNativeBackpropInput as unknown as KernelFunc
};
