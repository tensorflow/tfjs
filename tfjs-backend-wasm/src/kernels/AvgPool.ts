/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {backend_util, KernelFunc, NamedTensorInfoMap, registerKernel, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface AvgPoolInputs extends NamedTensorInfoMap {
  x: TensorInfo;
  filter: TensorInfo;
}

let wasmAvgPool: (
    xId: number, batchSize: number, inputHeight: number, inputWidth: number,
    filterHeight: number, filterWidth: number, padTop: number, padRight: number,
    padBottom: number, padLeft: number, strideHeight: number,
    strideWidth: number, channels: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmAvgPool = backend.wasm.cwrap('AvgPool', null /* void */, [
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
    'number',  // strideHeight
    'number',  // strideWidth
    'number',  // channels
    'number',  // outId
  ]);
}

function avgPool(args: {
  inputs: AvgPoolInputs,
  backend: BackendWasm,
  attrs: backend_util.Conv2DInfo
}) {
  const {inputs, attrs, backend} = args;
  const convInfo = attrs;

  const {x} = inputs;
  const xId = backend.dataIdMap.get(x.dataId).id;

  const filterHeight = convInfo.filterHeight;
  const filterWidth = convInfo.filterWidth;
  const padTop = convInfo.padInfo.top;
  const padRight = convInfo.padInfo.right;
  const padBottom = convInfo.padInfo.bottom;
  const padLeft = convInfo.padInfo.left;
  const strideHeight = convInfo.strideHeight;
  const strideWidth = convInfo.strideWidth;
  const channels = convInfo.inChannels;

  if (convInfo.dataFormat !== 'channelsLast') {
    throw new Error(
        `wasm backend does not support dataFormat:'` +
        `${convInfo.dataFormat}'. Please use 'channelsLast'.`);
  }

  if (convInfo.dilationWidth !== 1 || convInfo.dilationHeight !== 1) {
    throw new Error(
        `was backend only supports average pooling with dilation = [1, 1], ` +
        `got [${convInfo.dilationHeight}, ${convInfo.dilationWidth}].`);
  }

  const out = backend.makeOutput(convInfo.outShape, 'float32');
  const outId = backend.dataIdMap.get(out.dataId).id;

  wasmAvgPool(
      xId, x.shape[0], x.shape[1], x.shape[2], filterHeight, filterWidth,
      padTop, padRight, padBottom, padLeft, strideHeight, strideWidth, channels,
      outId);
  return out;
}

registerKernel({
  kernelName: 'AvgPool',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: avgPool as {} as KernelFunc
});
