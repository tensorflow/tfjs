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

import {KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';
import {Draw, DrawAttrs, DrawInputs,} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {DrawProgram} from '../draw_webgpu';

import {DrawTextureGraphicsProgram} from '../webgpu_graphics_program';

export function draw(
    args: {inputs: DrawInputs, backend: WebGPUBackend, attrs: DrawAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {image} = inputs;
  const {canvas, options} = attrs;
  const [height, width] = image.shape.slice(0, 2);
  const {imageOptions} = options || {};
  const alpha = imageOptions ?.alpha || 1;

  const format = 'rgba8unorm';
  const outShape = [height, width];
  const program = new DrawProgram(outShape, image.dtype, format);
  canvas.width = width;
  canvas.height = height;
  let gpuContext = canvas.getContext('webgpu');
  let canvasWebGPU;
  if (!gpuContext) {
    canvasWebGPU = document.createElement('canvas');
    gpuContext = canvasWebGPU.getContext('webgpu');
  }
  const numChannels = image.shape.length === 3 ? image.shape[2] : 1;
  //  'rgba8unorm' is not supported yet as the context format
  //  (https://bugs.chromium.org/p/chromium/issues/detail?id=1241369).
  //  If supported, we can use single compute pass to transfer the input tensor
  //  data to webgpu context canvas.
  gpuContext.configure({
    device: backend.device,
    format: 'bgra8unorm',
    alphaMode: 'premultiplied'
  });

  // DrawProgram first writes into a texture, then this texture will be
  // drawn into GPUCanvasContext.
  const outputDtype = 'int32';
  const usage = GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING;
  const output =
      backend.makeTensorInfoWithTexture(outShape, format, outputDtype, usage);
  const uniformData =
      [{type: 'uint32', data: [numChannels]}, {type: 'float32', data: [alpha]}];
  backend.runWebGPUProgram(program, [image], outputDtype, uniformData, output);

  const drawTextureProgram = new DrawTextureGraphicsProgram();
  backend.runWebGPUGraphicsProgram(drawTextureProgram, gpuContext, output);
  if (canvasWebGPU) {
    const canvas2dContext = canvas.getContext('2d');
    if (!canvas2dContext) {
      throw new Error(
          `Please make sure this canvas has only been used for 2d or webgpu context!`);
    }
    canvas2dContext.drawImage(canvasWebGPU, 0, 0);
  }
  backend.disposeData(output.dataId);
  return image;
}

export const drawConfig: KernelConfig = {
  kernelName: Draw,
  backendName: 'webgpu',
  kernelFunc: draw as unknown as KernelFunc
};
