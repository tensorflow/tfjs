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

import {FromPixelsAttrs, TensorInfo, util} from '@tensorflow/tfjs-core';
import {WebGPUBackend} from '../backend_webgpu';
import {FromPixelsProgram} from './FromPixels_utils/from_pixels_webgpu';
import * as webgpu_program from './webgpu_program';

export function fromPixelsImageBitmap(args: {
  imageBitmap: ImageBitmap,
  backend: WebGPUBackend,
  attrs: FromPixelsAttrs
}): TensorInfo {
  const {imageBitmap, backend, attrs} = args;
  const {numChannels} = attrs;

  const outShape = [imageBitmap.height, imageBitmap.width, numChannels];
  const size = util.sizeFromShape(outShape);
  const uniformData: [number, number] = [size, numChannels];

  const output = backend.makeTensorInfo(outShape, 'int32');
  if (!backend.fromPixelProgram) {
    backend.fromPixelProgram = new FromPixelsProgram(outShape);
  }

  // Different outShape will affect preprocessor result,
  // e.g. getCoordsFromFlatIndex. FromPixelsImageBitmap need
  // to recompile the pipeline to get the correct result.
  // FromPixelsImageBitmap leverages webgpu backend pipeline
  // cache system to avoid useless recompile.
  const outputShapes = [output.shape];
  const outputTypes = [output.dtype];
  const key = webgpu_program.makeShaderKey(
      backend.fromPixelProgram, outputShapes, outputTypes);

  const {bindGroupLayout, pipeline} = backend.getAndSavePipeline(key, () => {
    return webgpu_program.compileProgram(
        backend.glslang, backend.device, backend.fromPixelProgram, [], output);
  });
  backend.fromPixelProgram.setWebGPUBinary(bindGroupLayout, pipeline);

  backend.fromPixelProgram.updateOutputShape(outShape);

  backend.queue.copyImageBitmapToTexture(
      {imageBitmap, origin: {x: 0, y: 0}}, {
        texture: backend.fromPixelProgram.makeInputTexture(
            backend.device, imageBitmap.width, imageBitmap.height)
      },
      [imageBitmap.width, imageBitmap.height]);

  const info = backend.tensorMap.get(output.dataId);

  info.bufferInfo.buffer = backend.acquireBuffer(info.bufferInfo.byteSize);

  backend.fromPixelProgram.setUniform(backend.device, uniformData);

  backend.commandQueue.push(backend.fromPixelProgram.generateEncoder(
      backend.device, info.bufferInfo.buffer));
  backend.submitQueue();
  return output;
}
