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

import {backend_util, DepthwiseConv2dNative, DepthwiseConv2dNativeAttrs, DepthwiseConv2dNativeInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {DepthwiseConv2DProgram} from './depthwise_conv2d_webgpu';

export function depthwiseConv2dNative(args: {
  inputs: DepthwiseConv2dNativeInputs,
  attrs: DepthwiseConv2dNativeAttrs,
  backend: WebGPUBackend
}) {
  const {inputs, backend, attrs} = args;
  const {x, filter} = inputs;
  const {strides, pad, dilations, dimRoundingMode} = attrs;

  let $dilations = dilations;
  if ($dilations == null) {
    $dilations = [1, 1];
  }

  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number, number], strides, $dilations,
      pad, dimRoundingMode, true /* depthwise */);

  const program = new DepthwiseConv2DProgram(convInfo);
  const dimensions = [
    convInfo.filterHeight, convInfo.filterWidth, convInfo.padInfo.top,
    convInfo.padInfo.left, convInfo.strideHeight, convInfo.strideWidth,
    convInfo.dilationHeight, convInfo.dilationWidth, convInfo.inHeight,
    convInfo.inWidth
  ];
  const uniformData = new Int32Array(dimensions);
  return backend.runWebGPUProgram(program, [x, filter], x.dtype, uniformData);
}

export const depthwiseConv2dNativeConfig: KernelConfig = {
  kernelName: DepthwiseConv2dNative,
  backendName: 'webgpu',
  kernelFunc: depthwiseConv2dNative as {} as KernelFunc,
};
