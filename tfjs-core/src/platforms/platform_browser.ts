/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import {env} from '../environment';
import {getKernel} from '../kernel_registry';
import {tensor3d} from '../ops/tensor_ops';
import {Tensor3D} from '../tensor';

import {PixelData, Platform} from './platform';

export class PlatformBrowser implements Platform {
  private fromPixels2DContext: CanvasRenderingContext2D|
      OffscreenCanvasRenderingContext2D;

  constructor() {
    const canvas = createCanvas();
    if (canvas !== null) {
      this.fromPixels2DContext =
          canvas.getContext('2d') as CanvasRenderingContext2D;
    }
  }

  fromPixels(
      pixels: HTMLCanvasElement|PixelData|ImageData|HTMLImageElement|
      HTMLVideoElement,
      numChannels: number): Tensor3D {
    // Sanity checks.
    const isPixelData = (pixels as PixelData).data instanceof Uint8Array;
    const isImageData =
        typeof (ImageData) !== 'undefined' && pixels instanceof ImageData;
    const isVideo = typeof (HTMLVideoElement) !== 'undefined' &&
        pixels instanceof HTMLVideoElement;
    const isImage = typeof (HTMLImageElement) !== 'undefined' &&
        pixels instanceof HTMLImageElement;
    // tslint:disable-next-line:no-any
    const isCanvasLike = (pixels as any).getContext != null;
    if (!isCanvasLike && !isPixelData && !isImageData && !isVideo && !isImage) {
      throw new Error(
          'pixels passed to tf.browser.fromPixels() must be either an ' +
          `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData ` +
          `in browser, or OffscreenCanvas, ImageData in webworker` +
          ` or {data: Uint32Array, width: number, height: number}, ` +
          `but was ${(pixels as {}).constructor.name}`);
    }

    // If the current backend has 'FromPixels' registered, it has a more
    // efficient way of handling pixels uploads, so we call that.
    const kernel = getKernel('FromPixels', ENGINE.backendName);
    if (kernel != null) {
      return ENGINE.runKernel('FromPixels', {pixels} as {}, {numChannels}) as
          Tensor3D;
    }

    const [width, height] = isVideo ?
        [
          (pixels as HTMLVideoElement).videoWidth,
          (pixels as HTMLVideoElement).videoHeight
        ] :
        [pixels.width, pixels.height];
    let vals: Uint8ClampedArray|Uint8Array;

    if (isCanvasLike) {
      // tslint:disable-next-line:no-any
      vals = (pixels as any)
                 .getContext('2d')
                 .getImageData(0, 0, width, height)
                 .data;
    } else if (isImageData || isPixelData) {
      vals = (pixels as PixelData | ImageData).data;
    } else if (isImage || isVideo) {
      if (this.fromPixels2DContext == null) {
        throw new Error(
            'Can\'t read pixels from HTMLImageElement outside ' +
            'the browser.');
      }
      this.fromPixels2DContext.canvas.width = width;
      this.fromPixels2DContext.canvas.height = height;
      this.fromPixels2DContext.drawImage(
          pixels as HTMLVideoElement, 0, 0, width, height);
      vals = this.fromPixels2DContext.getImageData(0, 0, width, height).data;
    } else {
      throw new Error(
          'pixels passed to tf.browser.fromPixels() must be either an ' +
          `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData ` +
          `or {data: Uint8Array, width: number, height: number}, ` +
          `but was ${(pixels as {}).constructor.name}`);
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

  // According to the spec, the built-in encoder can do only UTF-8 encoding.
  // https://developer.mozilla.org/en-US/docs/Web/API/TextEncoder/TextEncoder
  private textEncoder: TextEncoder;

  fetch(path: string, init?: RequestInit): Promise<Response> {
    return fetch(path, init);
  }

  now(): number {
    return performance.now();
  }

  encode(text: string, encoding: string): Uint8Array {
    if (encoding !== 'utf-8' && encoding !== 'utf8') {
      throw new Error(
          `Browser's encoder only supports utf-8, but got ${encoding}`);
    }
    if (this.textEncoder == null) {
      this.textEncoder = new TextEncoder();
    }
    return this.textEncoder.encode(text);
  }
  decode(bytes: Uint8Array, encoding: string): string {
    return new TextDecoder(encoding).decode(bytes);
  }
}

function createCanvas() {
  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(300, 150);
  } else if (typeof document !== 'undefined') {
    return document.createElement('canvas');
  }
  return null;
}

if (env().get('IS_BROWSER')) {
  env().setPlatform('browser', new PlatformBrowser());
}
