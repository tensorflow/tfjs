/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {engine, FromPixels, FromPixelsAttrs, FromPixelsInputs, getKernel, NamedAttrMap, NamedTensorMap, PixelData, Tensor3D, tensor3d} from '@tensorflow/tfjs-core';

// tslint:disable-next-line: variable-name
export const FromPixelsAsync = 'FromPixelsAsync';

let fromPixels2DContext: CanvasRenderingContext2D;

function fromPixels_(
    pixels: PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
    HTMLVideoElement,
    numChannels = 3): Tensor3D {
  // Sanity checks.
  if (numChannels > 4) {
    throw new Error(
        'Cannot construct Tensor with more than 4 channels from pixels.');
  }
  if (pixels == null) {
    throw new Error('pixels passed to tf.browser.fromPixels() can not be null');
  }
  let isPixelData = false;
  let isImageData = false;
  let isVideo = false;
  let isImage = false;
  let isCanvasLike = false;
  if ((pixels as PixelData).data instanceof Uint8Array) {
    isPixelData = true;
  } else if (
      typeof (ImageData) !== 'undefined' && pixels instanceof ImageData) {
    isImageData = true;
  } else if (
      typeof (HTMLVideoElement) !== 'undefined' &&
      pixels instanceof HTMLVideoElement) {
    isVideo = true;
  } else if (
      typeof (HTMLImageElement) !== 'undefined' &&
      pixels instanceof HTMLImageElement) {
    isImage = true;
    // tslint:disable-next-line: no-any
  } else if ((pixels as any).getContext != null) {
    isCanvasLike = true;
  } else {
    throw new Error(
        'pixels passed to tf.browser.fromPixels() must be either an ' +
        `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData ` +
        `in browser, or OffscreenCanvas, ImageData in webworker` +
        ` or {data: Uint32Array, width: number, height: number}, ` +
        `but was ${(pixels as {}).constructor.name}`);
  }
  if (isVideo) {
    const HAVE_CURRENT_DATA_READY_STATE = 2;
    if (isVideo &&
        (pixels as HTMLVideoElement).readyState <
            HAVE_CURRENT_DATA_READY_STATE) {
      throw new Error(
          'The video element has not loaded data yet. Please wait for ' +
          '`loadeddata` event on the <video> element.');
    }
  }
  // If the current backend has 'FromPixels' registered, it has a more
  // efficient way of handling pixel uploads, so we call that.
  const kernel = getKernel(FromPixels, 'webgpu');
  if (kernel != null) {
    const inputs: FromPixelsInputs = {pixels};
    const attrs: FromPixelsAttrs = {numChannels};
    return engine().runKernel(
               FromPixels, inputs as {} as NamedTensorMap,
               attrs as {} as NamedAttrMap);
  }

  const [width, height] = isVideo ?
      [
        (pixels as HTMLVideoElement).videoWidth,
        (pixels as HTMLVideoElement).videoHeight
      ] :
      [pixels.width, pixels.height];
  let vals: Uint8ClampedArray|Uint8Array;

  if (isCanvasLike) {
    vals =
        // tslint:disable-next-line:no-any
        (pixels as any).getContext('2d').getImageData(0, 0, width, height).data;
  } else if (isImageData || isPixelData) {
    vals = (pixels as PixelData | ImageData).data;
  } else if (isImage || isVideo) {
    if (fromPixels2DContext == null) {
      fromPixels2DContext = document.createElement('canvas').getContext('2d');
    }
    fromPixels2DContext.canvas.width = width;
    fromPixels2DContext.canvas.height = height;
    fromPixels2DContext.drawImage(
        pixels as HTMLVideoElement, 0, 0, width, height);
    vals = fromPixels2DContext.getImageData(0, 0, width, height).data;
  }
  let values: Int32Array;
  if (numChannels === 4) {
    values = new Int32Array(vals);
  } else {
    const numPixels = width * height;
    values = new Int32Array(numPixels * numChannels);
    for (let i = 0; i < numPixels; i++) {
      for (let channel = 0; channel < numChannels; ++channel) {
        values[i * numChannels + channel] = vals[i * 4 + channel];
      }
    }
  }
  const outShape: [number, number, number] = [height, width, numChannels];
  return tensor3d(values, outShape, 'int32');
}

/**
 * Creates a `tf.Tensor` from an image in async way.
 *
 * ```js
 * const image = new ImageData(1, 1);
 * image.data[0] = 100;
 * image.data[1] = 150;
 * image.data[2] = 200;
 * image.data[3] = 255;
 *
 * (await tf.browser.fromPixelsAsync(image)).print();
 * ```
 *
 * @param pixels The input image to construct the tensor from. The
 * supported image types are all 4-channel. You can also pass in an image
 * object with following attributes:
 * `{data: Uint8Array; width: number; height: number}`
 * @param numChannels The number of channels of the output tensor. A
 * numChannels value less than 4 allows you to ignore channels. Defaults to
 * 3 (ignores alpha channel of input image).
 *
 * @doc {heading: 'Browser', namespace: 'browser', ignoreCI: true}
 */
export async function fromPixelsAsync(
    pixels: PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|
    HTMLVideoElement,
    numChannels = 3) {
  // Check whether the backend has FromPixelsAsycn kernel support,
  // if not fallback to normal fromPixels.
  // fromPixelAsync kernel doesn't support pixelData now, so fallback
  // to normal fromPixels.
  const kernel = getKernel(FromPixelsAsync, 'webgpu');
  if (kernel == null || (pixels as PixelData).data instanceof Uint8Array) {
    return fromPixels_(pixels, numChannels);
  }

  // Sanity checks.
  if (numChannels > 4) {
    throw new Error(
        'Cannot construct Tensor with more than 4 channels from pixels.');
  }
  if (pixels == null) {
    throw new Error('pixels passed to tf.browser.fromPixels() can not be null');
  }

  if (typeof (HTMLVideoElement) !== 'undefined' &&
      pixels instanceof HTMLVideoElement) {
    const HAVE_CURRENT_DATA_READY_STATE = 2;
    if (typeof (HTMLVideoElement) !== 'undefined' &&
        pixels instanceof HTMLVideoElement &&
        pixels.readyState < HAVE_CURRENT_DATA_READY_STATE) {
      throw new Error(
          'The video element has not loaded data yet. Please wait for ' +
          '`loadeddata` event on the <video> element.');
    }
  }

  const inputs: NamedTensorMap = {pixels} as {} as NamedTensorMap;
  const attrs: NamedAttrMap = {numChannels} as {} as NamedAttrMap;
  return await kernel.kernelFunc({inputs, attrs, backend: engine().backend}) as
      Tensor3D;
}
