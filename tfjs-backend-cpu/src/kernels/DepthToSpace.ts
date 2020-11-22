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

import {DepthToSpace, DepthToSpaceAttrs, DepthToSpaceInputs, KernelConfig, KernelFunc, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

export function depthToSpace(args: {
  inputs: DepthToSpaceInputs,
  backend: MathBackendCPU,
  attrs: DepthToSpaceAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {blockSize, dataFormat} = attrs;

  util.assert(
      dataFormat === 'NHWC',
      () => `Only NHWC dataFormat supported on CPU for depthToSpace. Got ${
          dataFormat}`);
  util.assert(
      blockSize > 1,
      () => `blockSize should be > 1 for depthToSpace, but was: ${blockSize}`);

  const batchSize = x.shape[0];
  const inputHeight = x.shape[1];
  const inputWidth = x.shape[2];
  const inputDepth = x.shape[3];

  const outputHeight = inputHeight * blockSize;
  const outputWidth = inputWidth * blockSize;
  const outputDepth = inputDepth / (blockSize * blockSize);

  const xValues = backend.data.get(x.dataId).values as TypedArray;
  const result =
      new Float32Array(batchSize * outputHeight * outputWidth * outputDepth);

  let outputIdx = 0;
  for (let b = 0; b < batchSize; ++b) {
    for (let h = 0; h < outputHeight; ++h) {
      const inH = Math.floor(h / blockSize);
      const offsetH = (h % blockSize);
      for (let w = 0; w < outputWidth; ++w) {
        const inW = Math.floor(w / blockSize);
        const offsetW = (w % blockSize);
        const offsetD = (offsetH * blockSize + offsetW) * outputDepth;
        for (let d = 0; d < outputDepth; ++d) {
          const inD = d + offsetD;
          const inputIdx =
              inD + inputDepth * (inW + inputWidth * (inH + inputHeight * b));
          result[outputIdx++] = xValues[inputIdx];
        }
      }
    }
  }

  return backend.makeTensorInfo(
      [batchSize, outputHeight, outputWidth, outputDepth], x.dtype, result);
}

export const depthToSpaceConfig: KernelConfig = {
  kernelName: DepthToSpace,
  backendName: 'cpu',
  kernelFunc: depthToSpace as {} as KernelFunc
};
