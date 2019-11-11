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

interface FusedConv2DInputs extends NamedTensorInfoMap {
  x: TensorInfo;
  filter: TensorInfo;
  bias?: TensorInfo;
}

let wasmFusedConv2d: (
    xId: number, batchSize: number, inputHeight: number, inputWidth: number,
    filterId: number, filterHeight: number, filterWidth: number, biasId: number,
    padTop: number, padRight: number, padBottom: number, padLeft: number,
    isSamePad: number, dilationHeight: number, dilationWidth: number,
    strideHeight: number, strideWidth: number, inputChannels: number,
    outputChannels: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmFusedConv2d = backend.wasm.cwrap('FusedConv2D', null /* void */, [
    'number',  // xId
    'number',  // batchSize
    'number',  // inputHeight
    'number',  // inputWidth
    'number',  // filterId
    'number',  // filterHeight
    'number',  // filterWidth
    'number',  // biasId
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

function fusedConv2d(args: {
  inputs: FusedConv2DInputs,
  backend: BackendWasm,
  attrs:
      {convInfo: backend_util.Conv2DInfo, activation: backend_util.Activation}
}) {
  const {inputs, attrs, backend} = args;
  const {convInfo, activation} = attrs;
  if (activation !== 'linear') {
    throw new Error(
        `${activation} activation not yet supported for FusedConv2D ` +
        `in the wasm backend.`);
  }

  const {x, filter, bias} = inputs;
  const xId = backend.dataIdMap.get(x.dataId).id;
  const filterId = backend.dataIdMap.get(filter.dataId).id;

  const outputChannels = convInfo.outChannels;

  let biasId = -1;
  if (bias != null) {
    const biasData = backend.dataIdMap.get(bias.dataId);
    if (biasData.shape.length !== 1) {
      throw new Error(
          `FusedConv2D only supports rank-1 bias but got ` +
          `rank ${biasData.shape.length}.`);
    }
    if (biasData.shape[0] !== outputChannels) {
      throw new Error(
          `FusedConv2D bias shape (${biasData.shape}) does not ` +
          `match the number of output channels (${outputChannels})`);
    }
    biasId = biasData.id;
  }

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
  const isSamePad = convInfo.padInfo.type === 'SAME' ? 1 : 0;
  const batchSize = convInfo.batchSize;
  const inHeight = convInfo.inHeight;
  const inWidth = convInfo.inWidth;

  if (convInfo.dataFormat !== 'channelsLast') {
    throw new Error(
        `wasm backend FusedConv2D does not support dataFormat:'` +
        `${convInfo.dataFormat}'. Please use 'channelsLast'.`);
  }

  const out = backend.makeOutput(convInfo.outShape, 'float32');
  const outId = backend.dataIdMap.get(out.dataId).id;
  wasmFusedConv2d(
      xId, batchSize, inHeight, inWidth, filterId, filterHeight, filterWidth,
      biasId, padTop, padRight, padBottom, padLeft, isSamePad, dilationHeight,
      dilationWidth, strideHeight, strideWidth, inputChannels, outputChannels,
      outId);
  return out;
}

registerKernel({
  kernelName: 'FusedConv2D',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: fusedConv2d as {} as KernelFunc
});
