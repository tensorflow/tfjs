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

import {backend_util, KernelConfig, KernelFunc, MaxPool, MaxPoolAttrs, MaxPoolInputs, Tensor4D} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmMaxPool: (
    xId: number, batchSize: number, inputHeight: number, inputWidth: number,
    filterHeight: number, filterWidth: number, padTop: number, padRight: number,
    padBottom: number, padLeft: number, dilationHeight: number,
    dilationWidth: number, strideHeight: number, strideWidth: number,
    inputChannels: number, outputChannels: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmMaxPool = backend.wasm.cwrap(MaxPool, null /* void */, [
    'number',  // xId
    'number',  // batchSize
    'number',  // inputHeight
    'number',  // inputWidth
    'number',  // filterHeight
    'number',  // filterWidth
    'number',  // padTop
    'number',  // padRight
    'number',  // padBottom
    'number',  // padLeft
    'number',  // dilationHeight
    'number',  // dilationWidth
    'number',  // strideHeight
    'number',  // strideWidth
    'number',  // inputChannels
    'number',  // outputChannels
    'number',  // outId
  ]);
}

function maxPool(
    args: {inputs: MaxPoolInputs, backend: BackendWasm, attrs: MaxPoolAttrs}) {
  const {inputs, attrs, backend} = args;

  const x = inputs.x as Tensor4D;
  const xId = backend.dataIdMap.get(x.dataId).id;

  const {filterSize, strides, pad, dimRoundingMode} = attrs;
  const convInfo = backend_util.computePool2DInfo(
      x.shape, filterSize, strides, 1 /* dilations */, pad, dimRoundingMode);

  const filterHeight = convInfo.filterHeight;
  const filterWidth = convInfo.filterWidth;
  const padTop = convInfo.padInfo.top;
  const padRight = convInfo.padInfo.right;
  const padBottom = convInfo.padInfo.bottom;
  const padLeft = convInfo.padInfo.left;
  const dilationHeight = convInfo.dilationHeight;
  const dilationWidth = convInfo.dilationWidth;
  const strideHeight = convInfo.strideHeight;
  const strideWidth = convInfo.strideWidth;
  const inputChannels = convInfo.inChannels;
  const outputChannels = convInfo.outChannels;

  if (convInfo.dataFormat !== 'channelsLast') {
    throw new Error(
        `wasm backend does not support dataFormat:'` +
        `${convInfo.dataFormat}'. Please use 'channelsLast'.`);
  }

  const out = backend.makeOutput(convInfo.outShape, 'float32');
  const outId = backend.dataIdMap.get(out.dataId).id;

  wasmMaxPool(
      xId, x.shape[0], x.shape[1], x.shape[2], filterHeight, filterWidth,
      padTop, padRight, padBottom, padLeft, dilationHeight, dilationWidth,
      strideHeight, strideWidth, inputChannels, outputChannels, outId);
  return out;
}

export const maxPoolConfig: KernelConfig = {
  kernelName: MaxPool,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: maxPool as {} as KernelFunc
};
