/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use backend file except in compliance with the License.
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

import {KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';
import {ToPixels, ToPixelsInputs, ToPixelsOutput} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {ToPixelsProgram} from '../to_pixels_webgpu';

export const toPixelsConfig: KernelConfig = {
  kernelName: ToPixels,
  backendName: 'webgpu',
  kernelFunc: toPixels as {} as KernelFunc,
};

export function toPixels(args: {
  inputs: ToPixelsInputs,
  backend: WebGPUBackend,
  attrs: ToPixelsOutput
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {$img} = inputs;
  const {canvas} = attrs;
  const [height, width] = $img.shape.slice(0, 2);

  const numChannels = 4;
  const outShape = [height, width, numChannels];
  const program = new ToPixelsProgram(outShape, $img.dtype);
  canvas.width = width;
  canvas.height = height;
  const gpuContext = canvas.getContext('webgpu');
  // 'rgba8unorm' is not supported yet as the context format. Otherwise, we
  // can save the second render pass. Ideally, just one comput pass, we can
  // transfer the input tensor data to webgpu context canvas and then return
  // the canvas to user. https://bugs.chromium.org/p/dawn/issues/detail?id=1219
  gpuContext.configure({
    device: backend.device,
    format: 'bgra8unorm',
    compositingAlphaMode: 'opaque'
  });

  const size = util.sizeFromShape(outShape);
  const strides = util.computeStrides(outShape);

  const uniformData = [
    {type: 'uint32', data: [size]}, {type: 'uint32', data: [numChannels]},
    {type: 'uint32', data: [...strides]}
  ];
  const outTensor = backend.runWebGPUProgram(
      program, [$img], 'int32', uniformData, null, gpuContext);

  return outTensor;
}
