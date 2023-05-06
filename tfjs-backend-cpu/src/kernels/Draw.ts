/**
 * @license
 * Copyright 2023 Google LLC.
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

import {Draw, DrawAttrs, DrawInputs, KernelConfig, KernelFunc, TypedArray} from '@tensorflow/tfjs-core';
import {TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export function draw(
    args: {inputs: DrawInputs, backend: MathBackendCPU, attrs: DrawAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {image} = inputs;
  const {canvas, options} = attrs;
  const {contextOptions, imageOptions} = options || {};
  const alpha = imageOptions ?.alpha || 1;

  const contextType = contextOptions ?.contextType || '2d';
  if (contextType !== '2d') {
    throw new Error(`Context type ${
        contextOptions.contextType} is not supported by the CPU backend.`);
  }
  const ctx = canvas.getContext(contextType,
    contextOptions?.contextAttributes || {}) as CanvasRenderingContext2D ;
  if (ctx == null) {
    throw new Error(`Could not get the context with ${contextType} type.`);
  }

  const [height, width] = image.shape.slice(0, 2);
  const depth = image.shape.length === 2 ? 1 : image.shape[2];
  const data = backend.data.get(image.dataId).values as TypedArray;
  const multiplier = image.dtype === 'float32' ? 255 : 1;
  const bytes = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < height * width; ++i) {
    const rgba = [0, 0, 0, 255 * alpha];

    for (let d = 0; d < depth; d++) {
      const value = data[i * depth + d];

      if (image.dtype === 'float32') {
        if (value < 0 || value > 1) {
          throw new Error(
              `Tensor values for a float32 Tensor must be in the ` +
              `range [0 - 1] but encountered ${value}.`);
        }
      } else if (image.dtype === 'int32') {
        if (value < 0 || value > 255) {
          throw new Error(
              `Tensor values for a int32 Tensor must be in the ` +
              `range [0 - 255] but encountered ${value}.`);
        }
      }

      if (depth === 1) {
        rgba[0] = value * multiplier;
        rgba[1] = value * multiplier;
        rgba[2] = value * multiplier;
      } else {
        rgba[d] = value * multiplier;
      }
    }

    const j = i * 4;
    bytes[j + 0] = Math.round(rgba[0]);
    bytes[j + 1] = Math.round(rgba[1]);
    bytes[j + 2] = Math.round(rgba[2]);
    bytes[j + 3] = Math.round(rgba[3]);
  }

  canvas.width = width;
  canvas.height = height;
  const imageData = new ImageData(bytes, width, height);
  ctx.putImageData(imageData, 0, 0);
  return image;
}

export const drawConfig: KernelConfig = {
  kernelName: Draw,
  backendName: 'cpu',
  kernelFunc: draw as unknown as KernelFunc
};
