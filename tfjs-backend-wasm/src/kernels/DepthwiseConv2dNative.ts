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

import {backend_util, DepthwiseConv2dNative, DepthwiseConv2dNativeAttrs, DepthwiseConv2dNativeInputs, KernelConfig, KernelFunc, Tensor4D} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmDepthwiseConv2d: (
    xId: number, batchSize: number, inputHeight: number, inputWidth: number,
    filterId: number, filterHeight: number, filterWidth: number, padTop: number,
    padRight: number, padBottom: number, padLeft: number, isSamePad: number,
    dilationHeight: number, dilationWidth: number, strideHeight: number,
    strideWidth: number, inputChannels: number, outputChannels: number,
    outId: number) => void;

function setup(backend: BackendWasm) {
  wasmDepthwiseConv2d =
      backend.wasm.cwrap(DepthwiseConv2dNative, null /* void */, [
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

function depthwiseConv2d(args: {
  inputs: DepthwiseConv2dNativeInputs,
  backend: BackendWasm,
  attrs: DepthwiseConv2dNativeAttrs
}) {
  const {inputs, attrs, backend} = args;

  const {x, filter} = inputs;
  const xId = backend.dataIdMap.get(x.dataId).id;
  const filterId = backend.dataIdMap.get(filter.dataId).id;

  const {strides, dilations, pad, dimRoundingMode} = attrs;

  const $dilations = dilations == null ? [1, 1] : dilations;

  const convInfo = backend_util.computeConv2DInfo(
      (x as Tensor4D).shape, (filter as Tensor4D).shape, strides,
      ($dilations as number | [number, number]), pad, dimRoundingMode,
      true /* depthwise */);

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
        `wasm backend DepthwiseConv2dNative does not support dataFormat:'` +
        `${convInfo.dataFormat}'. Please use 'channelsLast'.`);
  }

  const out = backend.makeOutput(convInfo.outShape, 'float32');
  const outId = backend.dataIdMap.get(out.dataId).id;
  wasmDepthwiseConv2d(
      xId, x.shape[0], x.shape[1], x.shape[2], filterId, filterHeight,
      filterWidth, padTop, padRight, padBottom, padLeft, isSamePad,
      dilationHeight, dilationWidth, strideHeight, strideWidth, inputChannels,
      outputChannels, outId);
  return out;
}

export const depthwiseConv2dNativeConfig: KernelConfig = {
  kernelName: DepthwiseConv2dNative,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: depthwiseConv2d as {} as KernelFunc
};
