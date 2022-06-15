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
import {FromPixels, FromPixelsAttrs, FromPixelsInputs, util} from '@tensorflow/tfjs-core';
import {backend_util, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {FromPixelsProgram} from '../from_pixels_webgpu';

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

  const isVideo = typeof (HTMLVideoElement) !== 'undefined' &&
      pixels instanceof HTMLVideoElement;
  const isImage = typeof (HTMLImageElement) !== 'undefined' &&
      pixels instanceof HTMLImageElement;
  const isCanvas = (typeof (HTMLCanvasElement) !== 'undefined' &&
                    pixels instanceof HTMLCanvasElement) ||
      (typeof (OffscreenCanvas) !== 'undefined' &&
       pixels instanceof OffscreenCanvas);
  const isImageBitmap =
      typeof (ImageBitmap) !== 'undefined' && pixels instanceof ImageBitmap;

  const [width, height] = isVideo ?
      [
        (pixels as HTMLVideoElement).videoWidth,
        (pixels as HTMLVideoElement).videoHeight
      ] :
      [pixels.width, pixels.height];
  const outShape = [height, width, numChannels];

  const useImport = env().getBool('WEBGPU_USE_IMPORT') && isVideo;
  const isVideoOrImage = isVideo || isImage;
  if (isImageBitmap || isCanvas || isVideoOrImage) {
    let externalImage;
    if (useImport) {
      externalImage = pixels as HTMLVideoElement;
    } else {
      if (isVideoOrImage) {
        if (fromPixels2DContext == null) {
          fromPixels2DContext =
              document.createElement('canvas').getContext('2d');
        }
        fromPixels2DContext.canvas.width = width;
        fromPixels2DContext.canvas.height = height;
        fromPixels2DContext.drawImage(
            pixels as HTMLVideoElement | HTMLImageElement, 0, 0, width, height);
        pixels = fromPixels2DContext.canvas;
      }
      externalImage = pixels as HTMLCanvasElement | ImageBitmap;
    }

    const size = util.sizeFromShape(outShape);
    const strides = util.computeStrides(outShape);
    const program = new FromPixelsProgram(outShape, numChannels, useImport);

    const uniformData = [
      {type: 'uint32', data: [size]}, {type: 'uint32', data: [numChannels]},
      {type: 'uint32', data: [...strides]}
    ];

    return backend.runFromPixelsProgram(
        program, outShape, uniformData, useImport, externalImage);
  }

  // TODO: Encoding should happen on GPU once we no longer have to download
  // image data to the CPU.
  const imageData = (pixels as ImageData | backend_util.PixelData).data;
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
