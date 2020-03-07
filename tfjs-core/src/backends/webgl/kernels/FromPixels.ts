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

import {env} from '../../../environment';
import {FromPixels, FromPixelsAttrs, FromPixelsInputs} from '../../../kernel_names';
import {KernelFunc, TensorInfo} from '../../../kernel_registry';
import {KernelConfig} from '../../../kernel_registry';
import {MathBackendWebGL} from '../backend_webgl';
import {TextureUsage} from '../tex_util';

import {FromPixelsProgram} from './FromPixels_utils/from_pixels_gpu';
import {FromPixelsPackedProgram} from './FromPixels_utils/from_pixels_packed_gpu';

export const fromPixelsConfig: KernelConfig = {
  kernelName: FromPixels,
  backendName: 'webgl',
  kernelFunc: fromPixels as {} as KernelFunc,
};

let fromPixels2DContext: CanvasRenderingContext2D;

function fromPixels(args: {
  inputs: FromPixelsInputs,
  backend: MathBackendWebGL,
  attrs: FromPixelsAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  let {pixels} = inputs;
  const {numChannels} = attrs;

  const isVideo = typeof (HTMLVideoElement) !== 'undefined' &&
      pixels instanceof HTMLVideoElement;
  const isImage = typeof (HTMLImageElement) !== 'undefined' &&
      pixels instanceof HTMLImageElement;
  const [width, height] = isVideo ?
      [
        (pixels as HTMLVideoElement).videoWidth,
        (pixels as HTMLVideoElement).videoHeight
      ] :
      [pixels.width, pixels.height];

  const texShape: [number, number] = [height, width];
  const outShape = [height, width, numChannels];

  if (isImage || isVideo) {
    if (fromPixels2DContext == null) {
      fromPixels2DContext = document.createElement('canvas').getContext('2d');
    }

    fromPixels2DContext.canvas.width = width;
    fromPixels2DContext.canvas.height = height;
    fromPixels2DContext.drawImage(
        pixels as HTMLVideoElement | HTMLImageElement, 0, 0, width, height);
    pixels = fromPixels2DContext.canvas;
  }

  const tempPixelHandle = backend.makeTensorInfo(texShape, 'int32');
  // This is a byte texture with pixels.
  backend.texData.get(tempPixelHandle.dataId).usage = TextureUsage.PIXELS;
  backend.gpgpu.uploadPixelDataToTexture(
      backend.getTexture(tempPixelHandle.dataId), pixels as ImageData);
  const program = env().getBool('WEBGL_PACK') ?
      new FromPixelsPackedProgram(outShape) :
      new FromPixelsProgram(outShape);
  const res = backend.runWebGLProgram(program, [tempPixelHandle], 'int32');
  backend.disposeData(tempPixelHandle.dataId);
  return res;
}
