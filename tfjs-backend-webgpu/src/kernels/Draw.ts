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

export function draw(
    args: {inputs: DrawInputs, backend: WebGPUBackend, attrs: DrawAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {image} = inputs;
  const {canvas, options} = attrs;
  const [height, width] = image.shape.slice(0, 2);
  const {imageOptions} = options || {};
  const alpha = imageOptions ?.alpha || 1;

  //  'rgba8unorm' should work on macOS according to
  //  https://bugs.chromium.org/p/chromium/issues/detail?id=1298618. But
  //  failed on macOS/M2. So use 'bgra8unorm' first when available.
  const format = backend.device.features.has('bgra8unorm-storage') ?
      'bgra8unorm' :
      'rgba8unorm';
  const outShape = [height, width];
  const program = new DrawProgram(outShape, image.dtype, format);
  canvas.width = width;
  canvas.height = height;
  const backendName = 'webgpu';
  let gpuContext = canvas.getContext(backendName);
  let canvasWebGPU;
  if (!gpuContext) {
    canvasWebGPU = new OffscreenCanvas(width, height);
    gpuContext = canvasWebGPU.getContext(backendName);
  }
  const numChannels = image.shape.length === 3 ? image.shape[2] : 1;
  gpuContext.configure({
    device: backend.device,
    format,
    usage: GPUTextureUsage.STORAGE_BINDING,
    alphaMode: 'premultiplied'
  });

  const outputDtype = 'int32';
  const output = backend.makeTensorInfo(outShape, outputDtype);
  const info = backend.tensorMap.get(output.dataId);
  info.resource = gpuContext.getCurrentTexture();
  info.external = true;

  const uniformData =
      [{type: 'uint32', data: [numChannels]}, {type: 'float32', data: [alpha]}];
  backend.runWebGPUProgram(program, [image], outputDtype, uniformData, output);

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
