/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {FromPixelsAttrs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {FromPixelsProgram} from '../from_pixels_webgpu';

type ExternalImage = HTMLCanvasElement|ImageBitmap|OffscreenCanvas;

export function fromPixelsExternalImage(args: {
  externalImage: ExternalImage|HTMLVideoElement,
  backend: WebGPUBackend,
  attrs: FromPixelsAttrs,
  outShape: number[],
  useImport: boolean
}): TensorInfo {
  const {externalImage, backend, attrs, outShape, useImport} = args;
  const {numChannels} = attrs;

  const size = util.sizeFromShape(outShape);
  const strides = util.computeStrides(outShape);
  const output = backend.makeTensorInfo(outShape, 'int32');
  const program = new FromPixelsProgram(outShape, useImport);

  const uniformData = [
    {type: 'uint32', data: [size]}, {type: 'uint32', data: [numChannels]},
    {type: 'uint32', data: [...strides]},
    {type: 'uint32', data: [...program.dispatch]}
  ];

  backend.runFromPixelsProgram(
      program, output, uniformData, useImport, externalImage);
  return output;
}
