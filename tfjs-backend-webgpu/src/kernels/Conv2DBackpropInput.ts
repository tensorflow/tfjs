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

import {backend_util, Conv2DBackpropInput, Conv2DBackpropInputAttrs, Conv2DBackpropInputInputs, env, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {Conv2DDerInputMMProgram} from '../conv_backprop_mm_webgpu';
import {Conv2DDerInputProgram} from '../conv_backprop_webgpu';

export function conv2DBackpropInput(args: {
  inputs: Conv2DBackpropInputInputs,
  attrs: Conv2DBackpropInputAttrs,
  backend: WebGPUBackend
}) {
  const {inputs, backend, attrs} = args;
  const {dy, filter} = inputs;
  const {inputShape, strides, pad, dataFormat, dimRoundingMode} = attrs;

  const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  const convInfo = backend_util.computeConv2DInfo(
      inputShape, filter.shape as [number, number, number, number], strides,
      1 /* dilations */, pad, dimRoundingMode, false, $dataFormat);

  const dimensions = [
    {type: 'int32', data: [convInfo.filterHeight, convInfo.filterWidth]},
    {
      type: 'int32',
      data: [
        convInfo.filterHeight - 1 - convInfo.padInfo.top,
        convInfo.filterWidth - 1 - convInfo.padInfo.left
      ]
    },
    {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]},
    {
      type: 'int32',
      data: [
        convInfo.batchSize, convInfo.outHeight, convInfo.outWidth,
        convInfo.outChannels
      ]
    },
  ];
  let program: Conv2DDerInputProgram|Conv2DDerInputMMProgram;
  if (env().getBool('WEBGPU_USE_NAIVE_CONV2D_TRANSPOSE')) {
    // Keep Conv2DDerInputProgram for reference.
    program = new Conv2DDerInputProgram(convInfo);
  } else {
    program = new Conv2DDerInputMMProgram(convInfo);
    const dimAOuter = convInfo.inShape[1] * convInfo.inShape[2];
    const dimBOuter = convInfo.inShape[3];
    const dimInner =
        convInfo.filterHeight * convInfo.filterWidth * convInfo.outChannels;
    dimensions.push(
        {type: 'uint32', data: [dimAOuter]},
        {type: 'uint32', data: [dimBOuter]},
        {type: 'uint32', data: [dimInner]});
  }
  return backend.runWebGPUProgram(program, [dy, filter], 'float32', dimensions);
}

export const conv2DBackpropInputConfig: KernelConfig = {
  kernelName: Conv2DBackpropInput,
  backendName: 'webgpu',
  kernelFunc: conv2DBackpropInput as {} as KernelFunc,
};
