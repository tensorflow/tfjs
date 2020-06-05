/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {ENGINE} from '../engine';
import {FromPixels, FromPixelsAttrs, FromPixelsInputs} from '../kernel_names';
import {getKernel, NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor2D, Tensor3D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {PixelData, TensorLike} from '../types';

import {op} from './operation';
import {tensor3d} from './tensor_ops';

let fromPixels2DContext: CanvasRenderingContext2D;

/**
 * Creates a `tf.Tensor` from an image.
 *
 * ```js
 * const image = new ImageData(1, 1);
 * image.data[0] = 100;
 * image.data[1] = 150;
 * image.data[2] = 200;
 * image.data[3] = 255;
 *
 * tf.browser.fromPixels(image).print();
 * ```
 *
 * @param pixels The input image to construct the tensor from. The
 * supported image types are all 4-channel. You can also pass in an image
 * object with following attributes:
 * `{data: Uint8Array; width: number; height: number}`
 * @param numChannels The number of channels of the output tensor. A
 * numChannels value less than 4 allows you to ignore channels. Defaults to
 * 3 (ignores alpha channel of input image).
 */
/** @doc {heading: 'Browser', namespace: 'browser', ignoreCI: true} */
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
  const kernel = getKernel(FromPixels, ENGINE.backendName);
  if (kernel != null) {
    const inputs: FromPixelsInputs = {pixels};
    const attrs: FromPixelsAttrs = {numChannels};
    return ENGINE.runKernel(
               FromPixels, inputs as {} as NamedTensorMap,
               attrs as {} as NamedAttrMap) as Tensor3D;
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
 * Draws a `tf.Tensor` of pixel values to a byte array or optionally a
 * canvas.
 *
 * When the dtype of the input is 'float32', we assume values in the range
 * [0-1]. Otherwise, when input is 'int32', we assume values in the range
 * [0-255].
 *
 * Returns a promise that resolves when the canvas has been drawn to.
 *
 * @param img A rank-2 or rank-3 tensor. If rank-2, draws grayscale. If
 *     rank-3, must have depth of 1, 3 or 4. When depth of 1, draws
 * grayscale. When depth of 3, we draw with the first three components of
 * the depth dimension corresponding to r, g, b and alpha = 1. When depth of
 * 4, all four components of the depth dimension correspond to r, g, b, a.
 * @param canvas The canvas to draw to.
 */
/** @doc {heading: 'Browser', namespace: 'browser'} */
export async function toPixels(
    img: Tensor2D|Tensor3D|TensorLike,
    canvas?: HTMLCanvasElement): Promise<Uint8ClampedArray> {
  let $img = convertToTensor(img, 'img', 'toPixels');
  if (!(img instanceof Tensor)) {
    // Assume int32 if user passed a native array.
    const originalImgTensor = $img;
    $img = originalImgTensor.toInt();
    originalImgTensor.dispose();
  }
  if ($img.rank !== 2 && $img.rank !== 3) {
    throw new Error(
        `toPixels only supports rank 2 or 3 tensors, got rank ${$img.rank}.`);
  }
  const [height, width] = $img.shape.slice(0, 2);
  const depth = $img.rank === 2 ? 1 : $img.shape[2];

  if (depth > 4 || depth === 2) {
    throw new Error(
        `toPixels only supports depth of size ` +
        `1, 3 or 4 but got ${depth}`);
  }

  const data = await $img.data();
  const minTensor = $img.min();
  const maxTensor = $img.max();
  const vals = await Promise.all([minTensor.data(), maxTensor.data()]);
  const minVals = vals[0];
  const maxVals = vals[1];
  const min = minVals[0];
  const max = maxVals[0];
  minTensor.dispose();
  maxTensor.dispose();
  if ($img.dtype === 'float32') {
    if (min < 0 || max > 1) {
      throw new Error(
          `Tensor values for a float32 Tensor must be in the ` +
          `range [0 - 1] but got range [${min} - ${max}].`);
    }
  } else if ($img.dtype === 'int32') {
    if (min < 0 || max > 255) {
      throw new Error(
          `Tensor values for a int32 Tensor must be in the ` +
          `range [0 - 255] but got range [${min} - ${max}].`);
    }
  } else {
    throw new Error(
        `Unsupported type for toPixels: ${$img.dtype}.` +
        ` Please use float32 or int32 tensors.`);
  }
  const multiplier = $img.dtype === 'float32' ? 255 : 1;
  const bytes = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < height * width; ++i) {
    let r, g, b, a;
    if (depth === 1) {
      r = data[i] * multiplier;
      g = data[i] * multiplier;
      b = data[i] * multiplier;
      a = 255;
    } else if (depth === 3) {
      r = data[i * 3] * multiplier;
      g = data[i * 3 + 1] * multiplier;
      b = data[i * 3 + 2] * multiplier;
      a = 255;
    } else if (depth === 4) {
      r = data[i * 4] * multiplier;
      g = data[i * 4 + 1] * multiplier;
      b = data[i * 4 + 2] * multiplier;
      a = data[i * 4 + 3] * multiplier;
    }

    const j = i * 4;
    bytes[j + 0] = Math.round(r);
    bytes[j + 1] = Math.round(g);
    bytes[j + 2] = Math.round(b);
    bytes[j + 3] = Math.round(a);
  }

  if (canvas != null) {
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(bytes, width, height);
    ctx.putImageData(imageData, 0, 0);
  }
  if ($img !== img) {
    $img.dispose();
  }
  return bytes;
}

export const fromPixels = op({fromPixels_});
