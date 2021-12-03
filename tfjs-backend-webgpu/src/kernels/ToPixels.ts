/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';
import {ToPixels, ToPixelsInputs, ToPixelsOutput} from '@tensorflow/tfjs-core';
import {TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
// import {ToPixelsProgram} from './toPixels_webgpu';
import {ToCanvasProgram} from './to_canvas_webgpu';

export const toPixelsConfig: KernelConfig = {
  kernelName: ToPixels,
  backendName: 'webgpu',
  kernelFunc: toPixels as {} as KernelFunc,
};

export function toPixels(args: {
  inputs: ToPixelsInputs,
  backend: WebGPUBackend,
  output: ToPixelsOutput
}): TensorInfo {
  const {inputs, backend /*, output*/} = args;
  let {$img} = inputs;
  // const {canvas} = output;
  const [height, width] = $img.shape.slice(0, 2);

  const outShape = [height, width, 4];
  const program = new ToCanvasProgram(outShape, $img.dtype);
  const gpuCanvas = document.createElement('canvas');
  gpuCanvas.width = width;
  gpuCanvas.height = height;
  const gpuContext = gpuCanvas.getContext('webgpu') as any;
  //  'rgba8unorm' is not supported yet as the context format. Otherwise, we
  //  can save the second render pass. Ideally, just one comput pass, we can
  //  transfer the input tensor data to webgpu context canvas and then return
  //  the canvas to user. https://bugs.chromium.org/p/dawn/issues/detail?id=1219
  gpuContext.configure({
    device: backend.device,
    format: 'bgra8unorm',
    compositingAlphaMode: 'opaque'
  });

  backend.runToCanvasProgram(program, $img, gpuContext);
  // In the final version, we should return a webgpu context canvas.
  // return gpuCanvas;

  // Currently, below codes are just used to test the correctness using the
  // existed toPixel tests. Once all faiures are resolved, I will do the
  // refacor, rename it to a new kernel, and return |gpuCanvas| directly.
  const testCanvas = document.createElement('canvas');
  const testContext = testCanvas.getContext('2d', {alpha: false});
  testContext.canvas.width = width;
  testContext.canvas.height = height;
  testContext.drawImage(gpuCanvas, 0, 0, width, height);
  const imageData = testContext.getImageData(0, 0, width, height).data;

  const outTensor = backend.makeTensorInfo(outShape, 'int32');
  const info = backend.tensorMap.get(outTensor.dataId);
  info.values = new Int32Array(imageData);
  backend.uploadToGPU(outTensor.dataId);
  return outTensor;
}
