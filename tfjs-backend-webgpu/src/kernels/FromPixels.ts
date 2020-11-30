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
import {backend_util, Tensor3D} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

var uint8ToFloat16Table = [
  0, 15360, 16384, 16896, 17408, 17664, 17920, 18176, 18432, 18560, 18688,
  18816, 18944, 19072, 19200, 19328, 19456, 19520, 19584, 19648, 19712,
  19776, 19840, 19904, 19968, 20032, 20096, 20160, 20224, 20288, 20352,
  20416, 20480, 20512, 20544, 20576, 20608, 20640, 20672, 20704, 20736,
  20768, 20800, 20832, 20864, 20896, 20928, 20960, 20992, 21024, 21056,
  21088, 21120, 21152, 21184, 21216, 21248, 21280, 21312, 21344, 21376,
  21408, 21440, 21472, 21504, 21520, 21536, 21552, 21568, 21584, 21600,
  21616, 21632, 21648, 21664, 21680, 21696, 21712, 21728, 21744, 21760,
  21776, 21792, 21808, 21824, 21840, 21856, 21872, 21888, 21904, 21920,
  21936, 21952, 21968, 21984, 22000, 22016, 22032, 22048, 22064, 22080,
  22096, 22112, 22128, 22144, 22160, 22176, 22192, 22208, 22224, 22240,
  22256, 22272, 22288, 22304, 22320, 22336, 22352, 22368, 22384, 22400,
  22416, 22432, 22448, 22464, 22480, 22496, 22512, 22528, 22536, 22544,
  22552, 22560, 22568, 22576, 22584, 22592, 22600, 22608, 22616, 22624,
  22632, 22640, 22648, 22656, 22664, 22672, 22680, 22688, 22696, 22704,
  22712, 22720, 22728, 22736, 22744, 22752, 22760, 22768, 22776, 22784,
  22792, 22800, 22808, 22816, 22824, 22832, 22840, 22848, 22856, 22864,
  22872, 22880, 22888, 22896, 22904, 22912, 22920, 22928, 22936, 22944,
  22952, 22960, 22968, 22976, 22984, 22992, 23000, 23008, 23016, 23024,
  23032, 23040, 23048, 23056, 23064, 23072, 23080, 23088, 23096, 23104,
  23112, 23120, 23128, 23136, 23144, 23152, 23160, 23168, 23176, 23184,
  23192, 23200, 23208, 23216, 23224, 23232, 23240, 23248, 23256, 23264,
  23272, 23280, 23288, 23296, 23304, 23312, 23320, 23328, 23336, 23344,
  23352, 23360, 23368, 23376, 23384, 23392, 23400, 23408, 23416, 23424,
  23432, 23440, 23448, 23456, 23464, 23472, 23480, 23488, 23496, 23504,
  23512, 23520, 23528, 23536, 23544
];

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
}): Tensor3D {
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

  if (env().getBool('FLOAT16') && env().getBool('DRIVER_SUPPORT_FLOAT16')) {
    let pixelFloat16Array = new Uint16Array(pixels.width * pixels.height * numChannels);
    const dataLength = imageData.length;
    let j = 0;
    if (numChannels != null && numChannels !== 4) {
      for (let i = 0; i < dataLength; i++) {
        if (i % 4 < numChannels) {
          pixelFloat16Array[j++] = uint8ToFloat16Table[imageData[i]];;
        }
      }
    } else {
      for (let i = 0; i < dataLength; i++) {
        pixelFloat16Array[j++] = uint8ToFloat16Table[imageData[i]];;
      }
    }

    const output = backend.makeOutputArray(outShape, 'float16');

    const info = backend.tensorMap.get(output.dataId);
    info.values = new Uint16Array(pixelFloat16Array);
    backend.maybeReleaseBuffer(output.dataId);

    backend.uploadToGPU(output.dataId);
    return output as Tensor3D;
  }

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
