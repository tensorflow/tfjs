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

import {backend_util, Conv2D, Conv2DAttrs, Conv2DInputs, KernelConfig, KernelFunc, Tensor4D} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmConv2d: (
    xId: number, batchSize: number, inputHeight: number, inputWidth: number,
    filterId: number, filterHeight: number, filterWidth: number, padTop: number,
    padRight: number, padBottom: number, padLeft: number, isSamePad: number,
    dilationHeight: number, dilationWidth: number, strideHeight: number,
    strideWidth: number, inputChannels: number, outputChannels: number,
    outId: number) => void;

function setup(backend: BackendWasm) {
  wasmConv2d = backend.wasm.cwrap(Conv2D, null /* void */, [
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
    'number',  // isSamePad
    'number',  // dilationHeight
    'number',  // dilationWidth
    'number',  // strideHeight
    'number',  // strideWidth
    'number',  // inputChannels
    'number',  // outputChannels
    'number',  // outId
  ]);
}

function conv2d(
    args: {inputs: Conv2DInputs, backend: BackendWasm, attrs: Conv2DAttrs}) {
  const {inputs, attrs, backend} = args;

  const {x, filter} = inputs;
  const xId = backend.dataIdMap.get(x.dataId).id;
  const filterId = backend.dataIdMap.get(filter.dataId).id;

  const {strides, dilations, pad, dimRoundingMode, dataFormat} = attrs;
  const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  const convInfo = backend_util.computeConv2DInfo(
      (x as Tensor4D).shape, (filter as Tensor4D).shape, strides, dilations,
      pad, dimRoundingMode, false, $dataFormat);

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
  const isSamePad = convInfo.padInfo.type === 'SAME' ? 1 : 0;

  if (convInfo.dataFormat !== 'channelsLast') {
    throw new Error(
        `wasm backend Conv2D does not support dataFormat:'` +
        `${convInfo.dataFormat}'. Please use 'channelsLast'.`);
  }

  const out = backend.makeOutput(convInfo.outShape, 'float32');
  const outId = backend.dataIdMap.get(out.dataId).id;
  wasmConv2d(
      xId, x.shape[0], x.shape[1], x.shape[2], filterId, filterHeight,
      filterWidth, padTop, padRight, padBottom, padLeft, isSamePad,
      dilationHeight, dilationWidth, strideHeight, strideWidth, inputChannels,
      outputChannels, outId);
  return out;
}

export const conv2DConfig: KernelConfig = {
  kernelName: Conv2D,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: conv2d as {} as KernelFunc
};
