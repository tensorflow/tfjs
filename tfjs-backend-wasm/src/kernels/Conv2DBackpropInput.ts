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

import {backend_util, Conv2DBackpropInput, Conv2DBackpropInputAttrs, Conv2DBackpropInputInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmConv2DBackpropInput: (
    dyId: number, filterId: number, batchSize: number, filterHeight: number,
    filterWidth: number, inHeight: number, inWidth: number, inChannels: number,
    outHeight: number, outWidth: number, outChannels: number,
    strideHeight: number, strideWidth: number, topPad: number, leftPad: number,
    fltS0: number, fltS1: number, fltS2: number, xBatchStride: number,
    xRowStride: number, xColStride: number, xChannelStride: number,
    yBatchStride: number, yRowStride: number, yColStride: number,
    yChannelStride: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmConv2DBackpropInput = backend.wasm.cwrap(Conv2DBackpropInput, null, [
    'number',  // dyId
    'number',  // filterId
    'number',  // batchSize
    'number',  // filterHeight
    'number',  // filterWidth
    'number',  // inHeight
    'number',  // inWidth
    'number',  // inChannels
    'number',  // outHeight
    'number',  // outWidth
    'number',  // outChannels
    'number',  // strideHeight
    'number',  // strideWidth
    'number',  // topPad
    'number',  // leftPad
    'number',  // fltS0
    'number',  // fltS1
    'number',  // fltS2
    'number',  // xBatchStride
    'number',  // xRowStride
    'number',  // xColStride
    'number',  // xChannelStride
    'number',  // yBatchStride
    'number',  // yRowStride
    'number',  // yColStride
    'number',  // yChannelStride
    'number',  // outId
  ]);
}

function conv2DBackpropInput(args: {
  backend: BackendWasm,
  inputs: Conv2DBackpropInputInputs,
  attrs: Conv2DBackpropInputAttrs
}): TensorInfo {
  const {backend, inputs, attrs} = args;
  const {dy, filter} = inputs;
  const {strides, pad, dataFormat, dimRoundingMode, inputShape} = attrs;

  const dilations = 1;

  const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  const convInfo = backend_util.computeConv2DInfo(
      inputShape, filter.shape as [number, number, number, number], strides,
      dilations, pad, dimRoundingMode, false /* depthwise */, $dataFormat);
  const {
    batchSize,
    filterHeight,
    filterWidth,
    inChannels,
    inHeight,
    inWidth,
    outChannels,
    outHeight,
    outWidth,
    strideHeight,
    strideWidth
  } = convInfo;

  const topPad = filterHeight - 1 - convInfo.padInfo.top;
  const leftPad = filterWidth - 1 - convInfo.padInfo.left;

  const isChannelsLast = convInfo.dataFormat === 'channelsLast';
  const dxStrides = util.computeStrides(convInfo.inShape);
  const dyStrides = util.computeStrides(dy.shape);
  const [fltS0, fltS1, fltS2] = util.computeStrides(filter.shape);
  const xBatchStride = dxStrides[0];
  const xRowStride = isChannelsLast ? dxStrides[1] : dxStrides[2];
  const xColStride = isChannelsLast ? dxStrides[2] : 1;
  const xChannelStride = isChannelsLast ? 1 : dxStrides[1];
  const yBatchStride = dyStrides[0];
  const yRowStride = isChannelsLast ? dyStrides[1] : dyStrides[2];
  const yColStride = isChannelsLast ? dyStrides[2] : 1;
  const yChannelStride = isChannelsLast ? 1 : dyStrides[1];

  const out = backend.makeOutput(convInfo.inShape, 'float32');
  const outId = backend.dataIdMap.get(out.dataId).id;
  const dyId = backend.dataIdMap.get(dy.dataId).id;
  const filterId = backend.dataIdMap.get(filter.dataId).id;

  wasmConv2DBackpropInput(
      dyId, filterId, batchSize, filterHeight, filterWidth, inHeight, inWidth,
      inChannels, outHeight, outWidth, outChannels, strideHeight, strideWidth,
      topPad, leftPad, fltS0, fltS1, fltS2, xBatchStride, xRowStride,
      xColStride, xChannelStride, yBatchStride, yRowStride, yColStride,
      yChannelStride, outId);
  return out;
}

export const conv2DBackpropInputConfig: KernelConfig = {
  kernelName: Conv2DBackpropInput,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: conv2DBackpropInput as {} as KernelFunc
};
