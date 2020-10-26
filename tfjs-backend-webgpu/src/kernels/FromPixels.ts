/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {env, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';
import {FromPixels, FromPixelsInputs, FromPixelsAttrs} from '@tensorflow/tfjs-core';
import {FromPixelsAsync} from '@tensorflow/tfjs-core';
import {backend_util, Tensor3D} from '@tensorflow/tfjs-core';
import {WebGPUBackend} from '../backend_webgpu';
import * as webgpu_program from './webgpu_program';
import {FromPixelsProgram} from './FromPixels_utils/from_pixels_webgpu';
import {util} from '@tensorflow/tfjs-core';

export const fromPixelsConfig: KernelConfig = {
  kernelName: FromPixels,
  backendName: 'webgpu',
  kernelFunc: fromPixels as {} as KernelFunc,
};

// Not many diffs with fromPixel, keep it here
export const fromPixelsAsyncConfig: KernelConfig = {
      kernelName: FromPixelsAsync,
      backendName: 'webgpu',
      kernelFunc: fromPixelsAsync as {} as KernelFunc,
    };

let fromPixels2DContext: CanvasRenderingContext2D;

async function fromPixelsAsync(args: {
      inputs: FromPixelsInputs,
      backend: WebGPUBackend,
      attrs: FromPixelsAttrs
    }) {
      const { inputs, backend, attrs } = args;
      const { pixels } = inputs;
      const { numChannels } = attrs;

      if (pixels == null) {
        throw new Error(
          'pixels passed to tf.browser.fromPixels() can not be null');
      }

      const outShape = [pixels.height, pixels.width, numChannels];
      const size = util.sizeFromShape(outShape);
      const uniformData: [number, number] = [size, numChannels];

      if (env().getBool('IS_BROWSER')) {
        // TODO: pixels.data instance of Uint8Array is not ImageBitmapSource,
        // This type should be handled in another shader.
        if (!(pixels instanceof HTMLVideoElement) &&
          !(pixels instanceof HTMLImageElement) &&
          !(pixels instanceof HTMLCanvasElement) &&
          !(pixels instanceof ImageData)) {
          throw new Error(
            'pixels passed to tf.browser.fromPixels() must be either an ' +
            `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData` +
            `but was ${(pixels as {}).constructor.name}`);
        }
      }

      // tslint:disable-next-line:no-any
      const imageBitmap = await createImageBitmap(pixels as any);

      const output = backend.makeOutputArray(outShape, 'int32');
      if (!backend.fromPixelProgram) {
        backend.fromPixelProgram = new FromPixelsProgram(outShape);

        const {bindGroupLayout, pipeline} = webgpu_program.compileProgram(
            backend.glslang,
            backend.device,
            backend.fromPixelProgram,
            [], output);

        backend.fromPixelProgram.setWebGPUBinary(bindGroupLayout, pipeline);
      }

      backend.queue.copyImageBitmapToTexture(
          {imageBitmap, origin: {x: 0, y: 0}}, {
            texture: backend.fromPixelProgram.makeInputTexture(
                backend.device, pixels.width, pixels.height)
          },
          {width: pixels.width, height: pixels.height, depth: 1});

      const info = backend.tensorMap.get(output.dataId);

      info.bufferInfo.buffer = backend.acquireBuffer(info.bufferInfo.byteSize);

      backend.fromPixelProgram.setUniform(backend.device, uniformData);

      backend.commandQueue.push(backend.fromPixelProgram.generateEncoder(
          backend.device, info.bufferInfo.buffer));
      backend.submitQueue();
      return output as Tensor3D;
  }

function fromPixels(args: {
  inputs: FromPixelsInputs,
  backend: WebGPUBackend,
  attrs: FromPixelsAttrs
}): Tensor3D {
  const { inputs, backend, attrs } = args;
  let { pixels } = inputs;
  const { numChannels } = attrs;

  if (pixels == null) {
    throw new Error(
      'pixels passed to tf.browser.fromPixels() can not be null');
  }

  const outShape = [pixels.height, pixels.width, numChannels];
  let imageData = (pixels as ImageData | backend_util.PixelData).data;

  if (env().getBool('IS_BROWSER')) {
    if (!(pixels instanceof HTMLVideoElement) &&
      !(pixels instanceof HTMLImageElement) &&
      !(pixels instanceof HTMLCanvasElement) &&
      !(pixels instanceof ImageData) &&
      !(pixels.data instanceof Uint8Array)) {
      throw new Error(
        'pixels passed to tf.browser.fromPixels() must be either an ' +
        `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData` +
        ` or {data: Uint32Array, width: number, height: number}, ` +
        `but was ${(pixels as {}).constructor.name}`);
    }
    if (pixels instanceof HTMLVideoElement ||
      pixels instanceof HTMLImageElement ||
      pixels instanceof HTMLCanvasElement) {
      if (fromPixels2DContext == null) {
        fromPixels2DContext =
          document.createElement('canvas').getContext('2d');
      }
      fromPixels2DContext.canvas.width = pixels.width;
      fromPixels2DContext.canvas.height = pixels.height;
      fromPixels2DContext.drawImage(
        pixels, 0, 0, pixels.width, pixels.height);
      pixels = fromPixels2DContext.canvas;
    }

    // TODO: Remove this once we figure out how to upload textures directly to
    // WebGPU.
    const imageDataLivesOnGPU = pixels instanceof HTMLVideoElement ||
      pixels instanceof HTMLImageElement ||
      pixels instanceof HTMLCanvasElement;
    if (imageDataLivesOnGPU) {
      imageData = fromPixels2DContext
        .getImageData(0, 0, pixels.width, pixels.height)
        .data;
    }
  }

  // TODO: Encoding should happen on GPU once we no longer have to download
  // image data to the CPU.
  let pixelArray = imageData;
  if (numChannels != null && numChannels !== 4) {
    pixelArray = new Uint8Array(pixels.width * pixels.height * numChannels);

    const dataLength = imageData.length;
    let j = 0;
    for (let i = 0; i < dataLength; i++) {
      if (i % 4 < numChannels) {
        pixelArray[j++] = imageData[i];
      }
    }
  }

  const output = backend.makeOutputArray(outShape, 'int32');

  const info = backend.tensorMap.get(output.dataId);
  info.values = new Int32Array(pixelArray);
  backend.maybeReleaseBuffer(output.dataId);

  backend.uploadToGPU(output.dataId);
  return output as Tensor3D;
}
