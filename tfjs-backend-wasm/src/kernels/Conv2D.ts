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

import {backend_util, NamedTensorInfoMap, registerKernel, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface Conv2DInputs extends NamedTensorInfoMap {
  x: TensorInfo;
  filter: TensorInfo;
}

let wasmConv2d: (
    xId: number, batchSize: number, inputHeight: number, inputWidth: number,
    filterId: number, filterHeight: number, filterWidth: number, padTop: number,
    padRight: number, padBottom: number, padLeft: number,
    dilationHeight: number, dilationWidth: number, strideHeight: number,
    strideWidth: number, inputChannels: number, outputChannels: number,
    outId: number) => void;

function setup(backend: BackendWasm) {
  wasmConv2d = backend.wasm.cwrap('Conv2D', null /* void */, [
    'number',  // xId
    'number',  // batchSize
    'number',  // inputHeight
    'number',  // inputWidth
    'number',  // filterId
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

function conv2d(args: {
  inputs: Conv2DInputs,
  backend: BackendWasm,
  attrs: backend_util.Conv2DInfo
}) {
  const {inputs, attrs, backend} = args;

  const {x, filter} = inputs;
  const xId = backend.dataIdMap.get(x.dataId).id;
  const filterId = backend.dataIdMap.get(filter.dataId).id;

  const filterHeight = attrs.filterHeight;
  const filterWidth = attrs.filterWidth;
  const padLeft = attrs.padInfo.left;
  const padTop = attrs.padInfo.top;
  const dilationHeight = attrs.dilationHeight;
  const dilationWidth = attrs.dilationWidth;
  const strideHeight = attrs.strideHeight;
  const strideWidth = attrs.strideWidth;

  if (attrs.dataFormat !== 'channelsLast') {
    throw new Error(
        `wasm backend does not support dataFormat:'` +
        `${attrs.dataFormat}'. Please use 'channelsLast'.`);
  }

  const out = backend.makeOutput(x.shape, 'float32');
  const outId = backend.dataIdMap.get(out.dataId).id;
  const xSize = util.sizeFromShape(x.shape);
  wasmConv2d(xId, x.shape[0], weightsId, outId);
  return out;
}

registerKernel({
  kernelName: 'Conv2D',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: conv2d
});
