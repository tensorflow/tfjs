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
import * as webgpu_program from './webgpu_program';

type ExternalImage = HTMLCanvasElement|ImageBitmap|OffscreenCanvas;

export function fromPixelsExternalImage(args: {
  externalImage: ExternalImage|HTMLVideoElement,
  backend: WebGPUBackend,
  attrs: FromPixelsAttrs,
  useImport: boolean
}): TensorInfo {
  const {externalImage, backend, attrs, useImport} = args;
  const {numChannels} = attrs;

  const outShape = [externalImage.height, externalImage.width, numChannels];
  const size = util.sizeFromShape(outShape);
  const strides = util.computeStrides(outShape);
  const uniformData = [size, numChannels, ...strides];
  const output = backend.makeTensorInfo(outShape, 'int32');
  const program =
      backend.getFromPixelsProgram(useImport ? 'import' : 'copyExternal');

  program.updateOutputShape(outShape);

  // Different outShape will affect preprocessor result,
  // e.g. getCoordsFromFlatIndex. FromPixelsImageExternalImage needs
  // to recompile the pipeline to get the correct result.
  // FromPixelsExternalImage leverages webgpu backend pipeline
  // cache system to avoid useless recompile.
  const outputShapes = [output.shape];
  const outputTypes = [output.dtype];
  const key = webgpu_program.makeShaderKey(program, outputShapes, outputTypes);

  const layout = program.getLayout(backend.device);

  const pipeline = backend.getAndSavePipeline(key, () => {
    return webgpu_program.compileProgram(
        backend.glslang, backend.device, program, layout.pipelineLayout, [],
        output, true);
  });

  program.setPipeline(pipeline);

  if (!useImport) {
    backend.queue.copyExternalImageToTexture(
        {source: externalImage as ExternalImage, origin: {x: 0, y: 0}}, {
          texture: program.makeInputTexture(
              backend.device, externalImage.width, externalImage.height)
        },
        [externalImage.width, externalImage.height]);
  }

  const info = backend.tensorMap.get(output.dataId);

  info.bufferInfo.buffer = backend.acquireBuffer(info.bufferInfo.byteSize);

  program.setUniform(backend.device, uniformData);

  let externalResource: GPUExternalTexture|GPUTextureView;
  if (useImport) {
    const externalTextureDescriptor = {
      source: externalImage as HTMLVideoElement
    };
    externalResource =
        backend.device.importExternalTexture(externalTextureDescriptor);
  } else {
    externalResource = program.inputTexture.createView();
  }

  backend.recordFromPixelsCommands(
      program, info.bufferInfo.buffer, layout, externalResource);
  backend.submitQueue();
  return output;
}
