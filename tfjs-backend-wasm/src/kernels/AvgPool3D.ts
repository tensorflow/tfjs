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

import {AvgPool3D, AvgPool3DAttrs, AvgPool3DInputs, backend_util, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmAvgPool3D: (
    xId: number, outId: number, batchSize: number, channelSize: number,
    inDepth: number, inHeight: number, inWidth: number, outDepth: number,
    outHeight: number, outWidth: number, strideDepth: number,
    strideHeight: number, strideWidth: number, dilationDepth: number,
    dilationHeight: number, dilationWidth: number, effectiveFilterDepth: number,
    effectiveFilterHeight: number, effectiveFilterWidth: number,
    padFront: number, padTop: number, padLeft: number) => void;

function setup(backend: BackendWasm) {
  wasmAvgPool3D = backend.wasm.cwrap('AvgPool3D', null, [
    'number',  // xId
    'number',  // outId
    'number',  // batchSize
    'number',  // channelSize
    'number',  // inDepth
    'number',  // inHeight
    'number',  // inWidth
    'number',  // outDepth
    'number',  // outHeight
    'number',  // outWidth
    'number',  // strideDepth
    'number',  // strideHeight
    'number',  // strideWidth
    'number',  // dilationDepth
    'number',  // dilationHeight
    'number',  // dilationWidth
    'number',  // effectiveFilterDepth
    'number',  // effectiveFilterHeight
    'number',  // effectiveFilterWidth
    'number',  // padFront
    'number',  // padTop
    'number',  // padLeft
  ]);
}

export function avgPool3D(args: {
  inputs: AvgPool3DInputs,
  attrs: AvgPool3DAttrs,
  backend: BackendWasm,
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {filterSize, strides, pad, dimRoundingMode, dataFormat} = attrs;

  const convInfo = backend_util.computePool3DInfo(
      x.shape as [number, number, number, number, number], filterSize, strides,
      /*dilations=*/1, pad, dimRoundingMode, dataFormat);
  const out = backend.makeOutput(convInfo.outShape, x.dtype);

  wasmAvgPool3D(
      backend.dataIdMap.get(x.dataId).id,
      backend.dataIdMap.get(out.dataId).id,
      convInfo.batchSize,
      // Since Pool3D ops (AvgPool3D and MaxPool3D) support 3D filter only, in
      // channels should always equal to out channels.
      /*channelSize=*/convInfo.inChannels,
      convInfo.inDepth,
      convInfo.inHeight,
      convInfo.inWidth,
      convInfo.outDepth,
      convInfo.outHeight,
      convInfo.outWidth,
      convInfo.strideDepth,
      convInfo.strideHeight,
      convInfo.strideWidth,
      convInfo.dilationDepth,
      convInfo.dilationHeight,
      convInfo.dilationWidth,
      convInfo.effectiveFilterDepth,
      convInfo.effectiveFilterHeight,
      convInfo.effectiveFilterWidth,
      convInfo.padInfo.front,
      convInfo.padInfo.top,
      convInfo.padInfo.left,
  );
  return out;
}

export const avgPool3DConfig: KernelConfig = {
  kernelName: AvgPool3D,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: avgPool3D as unknown as KernelFunc
};
