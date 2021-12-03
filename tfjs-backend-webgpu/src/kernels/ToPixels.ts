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

import {KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';
import {ToPixels, ToPixelsInputs, ToPixelsOutput} from '@tensorflow/tfjs-core';

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
  attrs: ToPixelsOutput
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {$img} = inputs;
  const {gpucanvas} = attrs;
  const [height, width] = $img.shape.slice(0, 2);

  const outShape = [height, width, 4];
  const program = new ToCanvasProgram(outShape, $img.dtype);
  gpucanvas.width = width;
  gpucanvas.height = height;
  const gpuContext = gpucanvas.getContext('webgpu');
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

  const outTensor = backend.makeTensorInfo(program.outputShape, 'int32');
  return outTensor;
}
