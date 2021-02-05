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
import {FromPixels, FromPixelsAttrs, FromPixelsInputs} from '@tensorflow/tfjs-core';
import {backend_util, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {fromPixelsImageBitmap} from './FromPixelsImageBitmap';

export const fromPixelsConfig: KernelConfig = {
  kernelName: FromPixels,
  backendName: 'webgpu',
  kernelFunc: fromPixels as {} as KernelFunc,
};

let fromPixels2DContext: CanvasRenderingContext2D;

export function fromPixels(args: {
  inputs: FromPixelsInputs,
  backend: WebGPUBackend,
  attrs: FromPixelsAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  let {pixels} = inputs;
  const {numChannels} = attrs;

  if (pixels == null) {
    throw new Error('pixels passed to tf.browser.fromPixels() can not be null');
  }

  const outShape = [pixels.height, pixels.width, numChannels];
  let imageData = (pixels as ImageData | backend_util.PixelData).data;

  if (env().getBool('IS_BROWSER')) {
    if (!(pixels instanceof HTMLVideoElement) &&
        !(pixels instanceof HTMLImageElement) &&
        !(pixels instanceof HTMLCanvasElement) &&
        !(pixels instanceof ImageData) && !(pixels instanceof ImageBitmap) &&
        !(pixels.data instanceof Uint8Array)) {
      throw new Error(
          'pixels passed to tf.browser.fromPixels() must be either an ' +
          `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData, ` +
          `ImageBitmap ` +
          `or {data: Uint32Array, width: number, height: number}, ` +
          `but was ${(pixels as {}).constructor.name}`);
    }

    if (pixels instanceof ImageBitmap) {
      return fromPixelsImageBitmap({imageBitmap: pixels, backend, attrs});
    }

    if (pixels instanceof HTMLVideoElement ||
        pixels instanceof HTMLImageElement ||
        pixels instanceof HTMLCanvasElement) {
      if (fromPixels2DContext == null) {
        fromPixels2DContext = document.createElement('canvas').getContext('2d');
      }
      fromPixels2DContext.canvas.width = pixels.width;
      fromPixels2DContext.canvas.height = pixels.height;
      fromPixels2DContext.drawImage(pixels, 0, 0, pixels.width, pixels.height);
      pixels = fromPixels2DContext.canvas;
    }

    // TODO: Remove this once we figure out how to upload textures directly to
    // WebGPU.
    const imageDataLivesOnGPU = pixels instanceof HTMLVideoElement ||
        pixels instanceof HTMLImageElement ||
        pixels instanceof HTMLCanvasElement;
    if (imageDataLivesOnGPU) {
      imageData =
          fromPixels2DContext.getImageData(0, 0, pixels.width, pixels.height)
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

  const output = backend.makeTensorInfo(outShape, 'int32');

  const info = backend.tensorMap.get(output.dataId);
  info.values = new Int32Array(pixelArray);
  backend.maybeReleaseBuffer(output.dataId);

  backend.uploadToGPU(output.dataId);
  return output;
}
