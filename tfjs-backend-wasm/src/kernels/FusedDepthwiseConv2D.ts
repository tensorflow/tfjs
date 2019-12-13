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

interface FusedDepthwiseConv2DInputs extends NamedTensorInfoMap {
  x: TensorInfo;
  filter: TensorInfo;
  bias?: TensorInfo;
}

let wasmFusedDepthwiseConv2d: (
    xId: number, batchSize: number, inputHeight: number, inputWidth: number,
    filterId: number, filterHeight: number, filterWidth: number, biasId: number,
    padTop: number, padRight: number, padBottom: number, padLeft: number,
    isSamePad: number, dilationHeight: number, dilationWidth: number,
    strideHeight: number, strideWidth: number, inputChannels: number,
    outputChannels: number, activation: number,
    preluActivationWeightsId: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmFusedDepthwiseConv2d =
      backend.wasm.cwrap('FusedDepthwiseConv2D', null /* void */, [
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
        'number',  // isSamePad
        'number',  // dilationHeight
        'number',  // dilationWidth
        'number',  // strideHeight
        'number',  // strideWidth
        'number',  // inputChannels
        'number',  // outputChannels
        'number',  // activation
        'number',  // preluActivationWeightsId
        'number',  // outId
      ]);
}

// Must match enum in conv2d_impl.h.
enum FusableActivation {
  linear = 0,
  relu = 1,
  relu6 = 2,
  prelu = 3
}

function fusedDepthwiseConv2d(args: {
  inputs: FusedDepthwiseConv2DInputs,
  backend: BackendWasm,
  attrs:
      {convInfo: backend_util.Conv2DInfo, activation: backend_util.Activation}
}) {
  const {inputs, attrs, backend} = args;
  const {convInfo, activation} = attrs;
  const fusedActivation =
      FusableActivation[activation as {} as keyof typeof FusableActivation];
  if (fusedActivation == null) {
    throw new Error(
        `${activation} activation not yet supported for FusedDepthwiseConv2D ` +
        `in the wasm backend.`);
  }

  const {x, filter, bias, preluActivationWeights} = inputs;
  const xId = backend.dataIdMap.get(x.dataId).id;
  const filterId = backend.dataIdMap.get(filter.dataId).id;

  const outputChannels = convInfo.outChannels;

  let biasId = 0;
  if (bias != null) {
    const biasData = backend.dataIdMap.get(bias.dataId);
    if (biasData.shape.length !== 1) {
      throw new Error(
          `FusedDepthwiseConv2D only supports rank-1 bias but got ` +
          `rank ${biasData.shape.length}.`);
    }
    if (biasData.shape[0] !== outputChannels) {
      throw new Error(
          `FusedDepthwiseConv2D bias shape (${biasData.shape}) does not ` +
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
        `wasm backend FusedDepthwiseConv2D does not support dataFormat:'` +
        `${convInfo.dataFormat}'. Please use 'channelsLast'.`);
  }

  const out = backend.makeOutput(convInfo.outShape, 'float32');
  const outId = backend.dataIdMap.get(out.dataId).id;
  const preluActivationWeightsId = preluActivationWeights == null ?
      0 :
      backend.dataIdMap.get(preluActivationWeights.dataId).id;
  wasmFusedDepthwiseConv2d(
      xId, batchSize, inHeight, inWidth, filterId, filterHeight, filterWidth,
      biasId, padTop, padRight, padBottom, padLeft, isSamePad, dilationHeight,
      dilationWidth, strideHeight, strideWidth, inputChannels, outputChannels,
      fusedActivation, preluActivationWeightsId, outId);
  return out;
}

registerKernel({
  kernelName: 'FusedDepthwiseConv2D',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: fusedDepthwiseConv2d as {} as KernelFunc
});
