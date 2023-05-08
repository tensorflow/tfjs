/**
 * @license
 * Copyright 2023 Google LLC.
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

import {backend_util, Conv3DBackpropInputV2, Conv3DBackpropInputV2Attrs, Conv3DBackpropInputV2Inputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmConv3DBackpropInputV2: (
    filterId: number, dyId: number, dxId: number, batchSize: number,
    inDepth: number, inHeight: number, inWidth: number, inChannels: number,
    outDepth: number, outHeight: number, outWidth: number, outChannels: number,
    strideDepth: number, strideHeight: number, strideWidth: number,
    dilationDepth: number, dilationHeight: number, dilationWidth: number,
    filterDepth: number, filterHeight: number, filterWidth: number,
    padFront: number, padTop: number, padLeft: number) => void;

function setup(backend: BackendWasm) {
  wasmConv3DBackpropInputV2 = backend.wasm.cwrap(Conv3DBackpropInputV2, null, [
    'number',  // filterId
    'number',  // dyId
    'number',  // dxId
    'number',  // batchSize
    'number',  // inDepth
    'number',  // inHeight
    'number',  // inWidth
    'number',  // inChannels
    'number',  // outDepth
    'number',  // outHeight
    'number',  // outWidth
    'number',  // outChannels
    'number',  // strideDepth
    'number',  // strideHeight
    'number',  // strideWidth
    'number',  // dilationDepth
    'number',  // dilationHeight
    'number',  // dilationWidth
    'number',  // filterDepth
    'number',  // filterHeight
    'number',  // filterWidth
    'number',  // padFront
    'number',  // padTop
    'number',  // padLeft
  ]);
}

export function conv3DBackpropInputV2(args: {
  inputs: Conv3DBackpropInputV2Inputs,
  attrs: Conv3DBackpropInputV2Attrs,
  backend: BackendWasm,
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, filter} = inputs;
  const {pad, strides, inputShape} = attrs;
  if (dy.dtype !== 'float32') {
    throw new Error(`Tensor dy must have dtype float32, got ${dy.dtype}`);
  }
  if (filter.dtype !== 'float32') {
    throw new Error(
        `Tensor filter must have dtype float32, got ${filter.dtype}`);
  }

  const convInfo = backend_util.computeConv3DInfo(
      inputShape, filter.shape as [number, number, number, number, number],
      strides, /*dilations=*/1, pad);

  const dx = backend.makeOutput(convInfo.inShape, dy.dtype);

  wasmConv3DBackpropInputV2(
      backend.dataIdMap.get(filter.dataId).id,
      backend.dataIdMap.get(dy.dataId).id,
      backend.dataIdMap.get(dx.dataId).id,
      convInfo.batchSize,
      convInfo.inDepth,
      convInfo.inHeight,
      convInfo.inWidth,
      convInfo.inChannels,
      convInfo.outDepth,
      convInfo.outHeight,
      convInfo.outWidth,
      convInfo.outChannels,
      convInfo.strideDepth,
      convInfo.strideHeight,
      convInfo.strideWidth,
      convInfo.dilationDepth,
      convInfo.dilationHeight,
      convInfo.dilationWidth,
      convInfo.filterDepth,
      convInfo.filterHeight,
      convInfo.filterWidth,
      convInfo.padInfo.front,
      convInfo.padInfo.top,
      convInfo.padInfo.left,
  );
  return dx;
}

export const conv3DBackpropInputV2Config: KernelConfig = {
  kernelName: Conv3DBackpropInputV2,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: conv3DBackpropInputV2 as unknown as KernelFunc
};
