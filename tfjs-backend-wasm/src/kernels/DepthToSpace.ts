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

import {DepthToSpace, DepthToSpaceAttrs, DepthToSpaceInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmDepthToSpace: (
    xId: number, blockSize: number, channelsLast: number, xStrides: Uint8Array,
    xStridesLength: number, outputShape: Uint8Array, outputStrides: Uint8Array,
    outSize: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmDepthToSpace = backend.wasm.cwrap(DepthToSpace, null /*void*/, [
    'number',  // xId
    'number',  // blockSize
    'number',  // channelsLast
    'array',   // xStrides
    'number',  // xStridesLength
    'array',   // outputShape
    'array',   // outputStrides
    'number',  // outSize
    'number',  // outId
  ]);
}

export function depthToSpace(args: {
  backend: BackendWasm,
  inputs: DepthToSpaceInputs,
  attrs: DepthToSpaceAttrs
}): TensorInfo {
  const {backend, inputs, attrs} = args;
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

  const out = backend.makeOutput(outputShape, 'float32');

  const xData = backend.dataIdMap.get(x.dataId);
  const xId = xData.id;
  const xStridesBytes =
      new Uint8Array(new Int32Array(util.computeStrides(x.shape)).buffer);

  const outputShapeBytes = new Uint8Array(new Int32Array(outputShape).buffer);
  const outStridesBytes =
      new Uint8Array(new Int32Array(util.computeStrides(outputShape)).buffer);

  const outId = backend.dataIdMap.get(out.dataId).id;
  const channelsLast = dataFormat === 'NHWC' ? 1 : 0;
  wasmDepthToSpace(
      xId, blockSize, channelsLast, xStridesBytes, x.shape.length - 1,
      outputShapeBytes, outStridesBytes, outputShape.length, outId);

  return out;
}

export const depthToSpaceConfig: KernelConfig = {
  kernelName: DepthToSpace,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: depthToSpace as {} as KernelFunc
};
