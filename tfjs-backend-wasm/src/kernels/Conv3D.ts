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

import {backend_util, Conv3D, Conv3DAttrs, Conv3DInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmConv3D: (
    xId: number, filterId: number, outId: number, batchSize: number,
    inDepth: number, inHeight: number, inWidth: number, inChannels: number,
    outDepth: number, outHeight: number, outWidth: number, outChannels: number,
    strideDepth: number, strideHeight: number, strideWidth: number,
    dilationDepth: number, dilationHeight: number, dilationWidth: number,
    filterDepth: number, filterHeight: number, filterWidth: number,
    padFront: number, padTop: number, padLeft: number) => void;

function setup(backend: BackendWasm) {
  wasmConv3D = backend.wasm.cwrap(Conv3D, null, [
    'number',  // xId
    'number',  // filterId
    'number',  // outId
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

export function conv3D(args: {
  inputs: Conv3DInputs,
  attrs: Conv3DAttrs,
  backend: BackendWasm,
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, filter} = inputs;
  const {strides, pad, dilations} = attrs;
  if (x.dtype !== 'float32') {
    throw new Error(`Tensor x must have dtype float32, got ${x.dtype}`);
  }
  if (filter.dtype !== 'float32') {
    throw new Error(
        `Tensor filter must have dtype float32, got ${filter.dtype}`);
  }

  const convInfo = backend_util.computeConv3DInfo(
      x.shape as [number, number, number, number, number],
      filter.shape as [number, number, number, number, number], strides,
      dilations, pad);

  const out = backend.makeOutput(convInfo.outShape, x.dtype);
  wasmConv3D(
      backend.dataIdMap.get(x.dataId).id,
      backend.dataIdMap.get(filter.dataId).id,
      backend.dataIdMap.get(out.dataId).id,
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
  return out;
}

export const conv3DConfig: KernelConfig = {
  kernelName: Conv3D,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: conv3D as unknown as KernelFunc
};
