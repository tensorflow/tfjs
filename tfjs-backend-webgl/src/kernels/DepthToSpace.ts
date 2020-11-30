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

import {DepthToSpace, DepthToSpaceAttrs, DepthToSpaceInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {DepthToSpaceProgram} from '../depth_to_space_gpu';

export function depthToSpace(args: {
  inputs: DepthToSpaceInputs,
  backend: MathBackendWebGL,
  attrs: DepthToSpaceAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {blockSize, dataFormat} = attrs;

  util.assert(
      blockSize > 1,
      () => `blockSize should be > 1 for depthToSpace, but was: ${blockSize}`);

  const batchSize = x.shape[0];
  const inputHeight = (dataFormat === 'NHWC') ? x.shape[1] : x.shape[2];
  const inputWidth = (dataFormat === 'NHWC') ? x.shape[2] : x.shape[3];
  const inputDepth = (dataFormat === 'NHWC') ? x.shape[3] : x.shape[1];

  const outputHeight = inputHeight * blockSize;
  const outputWidth = inputWidth * blockSize;
  const outputDepth = inputDepth / (blockSize * blockSize);

  const outputShape = (dataFormat === 'NHWC') ?
      [batchSize, outputHeight, outputWidth, outputDepth] :
      [batchSize, outputDepth, outputHeight, outputWidth];

  const program = new DepthToSpaceProgram(outputShape, blockSize, dataFormat);
  return backend.runWebGLProgram(program, [x], x.dtype);
}

export const depthToSpaceConfig: KernelConfig = {
  kernelName: DepthToSpace,
  backendName: 'webgl',
  kernelFunc: depthToSpace as {} as KernelFunc
};
