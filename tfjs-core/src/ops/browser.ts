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
import {Tensor, Tensor2D, Tensor3D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {PixelData, TensorLike} from '../types';

import {op} from './operation';

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
  if (numChannels > 4) {
    throw new Error(
        'Cannot construct Tensor with more than 4 channels from pixels.');
  }
  return ENGINE.fromPixels(pixels, numChannels);
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
    $img = $img.toInt();
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
  const [minVals, maxVals] =
      await Promise.all([minTensor.data(), maxTensor.data()]);
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
