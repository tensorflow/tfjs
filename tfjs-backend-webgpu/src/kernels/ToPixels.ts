/**
 * @license
 * Copyright 2023 Google LLC.
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
import {ToPixels, ToPixelsAttrs, ToPixelsInputs} from '@tensorflow/tfjs-core';

import {TextureInfo, WebGPUBackend} from '../backend_webgpu';
import {ToPixelsProgram} from '../to_pixels_webgpu';

export const toPixelsConfig: KernelConfig = {
  kernelName: ToPixels,
  backendName: 'webgpu',
  kernelFunc: toPixels as {} as KernelFunc,
};

function getNumberChannel(format: GPUTextureFormat) {
  if (format === 'rgba8unorm' || format === 'bgra8unorm') {
    return 4;
  } else {
    throw new Error(`${format} is not supported!`);
  }
}

export function toPixels(
    args:
        {inputs: ToPixelsInputs, backend: WebGPUBackend, attrs: ToPixelsAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {$img} = inputs;
  const {canvas, numChannels} = attrs;
  const [height, width] = $img.shape.slice(0, 2);

  const format = 'rgba8unorm';
  const outShape = [height, width, getNumberChannel(format)];
  const program = new ToPixelsProgram(outShape, $img.dtype, format);
  canvas.width = width;
  canvas.height = height;
  const gpuContext = canvas.getContext('webgpu');
  //  'rgba8unorm' is not supported yet as the context format
  //  (https://bugs.chromium.org/p/chromium/issues/detail?id=1241369).
  //  If supported, we can use single compute pass to transfer the input tensor
  //  data to webgpu context canvas.
  gpuContext.configure({
    device: backend.device,
    format: 'bgra8unorm',
    compositingAlphaMode: 'opaque'
  });

  // ToPixelsProgram will frist write this texture, then this texture will be
  // drawn into GPUCanvasContext.
  const intermediateTexture = backend.device.createTexture({
    format,
    size: [width, height, 1],
    usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING
  });

  const textureInfo: TextureInfo = {
    width,
    height,
    format: null,
    usage: null,
    texture: intermediateTexture,
    isCanvas: true
  };

  const size = util.sizeFromShape(program.outputShape);
  const strides = util.computeStrides(program.outputShape);

  const uniformData = [
    {type: 'uint32', data: [size]}, {type: 'uint32', data: [numChannels]},
    {type: 'uint32', data: [...strides]}
  ];

  return backend.runWebGPUProgram(
      program, [$img], 'int32', uniformData, null, textureInfo, gpuContext);
}
