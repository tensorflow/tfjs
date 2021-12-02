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
  /*
  const depth = $img.shape.length === 2 ? 1 : $img.shape[2];
  const multiplier = $img.dtype === 'float32' ? 255 : 1;
  const outShape = [height, width, 4];
  const program = new ToPixelsProgram(outShape);
  const uniformData =
      [{type: 'float32', data: [multiplier]}, {type: 'int32', data: [depth]}];
  const pixels =
      backend.runWebGPUProgram(program, [$img], 'int32', uniformData);
  return pixels;
*/
  const outShape = [height, width, 4];
  const program = new ToCanvasProgram(outShape, $img.dtype);
  const newCanvas = document.createElement('canvas');
  backend.runToCanvasProgram(program, $img, newCanvas);
  const ctx = newCanvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, width, height).data;

  const outTensor = backend.makeTensorInfo(outShape, 'int32');
  const info = backend.tensorMap.get(outTensor.dataId);
  info.values = new Int32Array(imageData);
  backend.uploadToGPU(outTensor.dataId);
  return outTensor;
  // upload pixels to a temporary webgpu canvas by canvas->getCurrentTexture =>
  // copyBufferToTexture. Then if canvas is a 2d canvas, then call
  // 2dCanvasContext.drawImage If canvas is a webgl canvas,
  // texImage2D(webgpuCanvas)->texture, render the texture to webgl canvas. If
  // cavas is a webgpu canvas, copyExternalImageToTexture.
}
