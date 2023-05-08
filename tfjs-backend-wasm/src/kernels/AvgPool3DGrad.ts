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

import {AvgPool3DGrad, AvgPool3DGradAttrs, AvgPool3DGradInputs, backend_util, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmAvgPool3DGrad: (
    dyId: number, dxId: number, batchSize: number, channelSize: number,
    inDepth: number, inHeight: number, inWidth: number, outDepth: number,
    outHeight: number, outWidth: number, strideDepth: number,
    strideHeight: number, strideWidth: number, dilationDepth: number,
    dilationHeight: number, dilationWidth: number, effectiveFilterDepth: number,
    effectiveFilterHeight: number, effectiveFilterWidth: number,
    padFront: number, padTop: number, padLeft: number, filterDepth: number,
    filterHeight: number, filterWidth: number) => void;

function setup(backend: BackendWasm) {
  wasmAvgPool3DGrad = backend.wasm.cwrap('AvgPool3DGrad', null, [
    'number',  // dyId
    'number',  // dxId
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
    'number',  // filterDepth
    'number',  // filterHeight
    'number',  // filterWidth
  ]);
}

export function avgPool3DGrad(args: {
  inputs: AvgPool3DGradInputs,
  attrs: AvgPool3DGradAttrs,
  backend: BackendWasm,
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, input} = inputs;
  const {filterSize, strides, pad, dimRoundingMode} = attrs;

  const convInfo = backend_util.computePool3DInfo(
      input.shape as [number, number, number, number, number], filterSize,
      strides, /*dilations=*/1, pad, dimRoundingMode);
  const dx = backend.makeOutput(input.shape, input.dtype);

  wasmAvgPool3DGrad(
      backend.dataIdMap.get(dy.dataId).id,
      backend.dataIdMap.get(dx.dataId).id,
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
      convInfo.filterDepth,
      convInfo.filterHeight,
      convInfo.filterWidth,
  );
  return dx;
}

export const avgPool3DGradConfig: KernelConfig = {
  kernelName: AvgPool3DGrad,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: avgPool3DGrad as unknown as KernelFunc
};
