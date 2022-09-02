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

import {backend_util, Conv2D, Conv2DAttrs, Conv2DInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {conv2DImpl} from './Conv2D_impl';

export function conv2d(
    args: {inputs: Conv2DInputs, attrs: Conv2DAttrs, backend: WebGPUBackend}) {
  const {inputs, attrs, backend} = args;
  const {x, filter} = inputs;
  const {strides, pad, dataFormat, dilations, dimRoundingMode} = attrs;
  const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number, number], strides, dilations, pad,
      dimRoundingMode, false /* depthwise */, $dataFormat);
  return conv2DImpl({x, filter, convInfo, backend});
}

export const conv2DConfig: KernelConfig = {
  kernelName: Conv2D,
  backendName: 'webgpu',
  kernelFunc: conv2d as {} as KernelFunc
};
