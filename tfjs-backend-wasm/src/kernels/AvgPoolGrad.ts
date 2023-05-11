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

import {AvgPoolGrad, AvgPoolGradAttrs, AvgPoolGradInputs, backend_util, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmAvgPoolGrad: (
    dyId: number, dxId: number, batchSize: number, channelSize: number,
    inHeight: number, inWidth: number, outHeight: number, outWidth: number,
    strideHeight: number, strideWidth: number, dilationHeight: number,
    dilationWidth: number, effectiveFilterHeight: number,
    effectiveFilterWidth: number, padTop: number, padLeft: number,
    filterHeight: number, filterWidth: number) => void;

function setup(backend: BackendWasm) {
  wasmAvgPoolGrad = backend.wasm.cwrap('AvgPoolGrad', null, [
    'number',  // dyId
    'number',  // dxId
    'number',  // batchSize
    'number',  // channelSize
    'number',  // inHeight
    'number',  // inWidth
    'number',  // outHeight
    'number',  // outWidth
    'number',  // strideHeight
    'number',  // strideWidth
    'number',  // dilationHeight
    'number',  // dilationWidth
    'number',  // effectiveFilterHeight
    'number',  // effectiveFilterWidth
    'number',  // padTop
    'number',  // padLeft
    'number',  // filterHeight
    'number',  // filterWidth
  ]);
}

export function avgPoolGrad(args: {
  inputs: AvgPoolGradInputs,
  attrs: AvgPoolGradAttrs,
  backend: BackendWasm,
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, input} = inputs;
  const {filterSize, strides, pad} = attrs;

  const convInfo = backend_util.computePool2DInfo(
      input.shape as [number, number, number, number], filterSize, strides,
      /*dilations=*/1, pad);
  const dx = backend.makeOutput(input.shape, input.dtype);

  wasmAvgPoolGrad(
      backend.dataIdMap.get(dy.dataId).id,
      backend.dataIdMap.get(dx.dataId).id,
      convInfo.batchSize,
      // Since Pool ops (AvgPool and MaxPool) support 2D filter only, in
      // channels should always equal to out channels.
      /*channelSize=*/convInfo.inChannels,
      convInfo.inHeight,
      convInfo.inWidth,
      convInfo.outHeight,
      convInfo.outWidth,
      convInfo.strideHeight,
      convInfo.strideWidth,
      convInfo.dilationHeight,
      convInfo.dilationWidth,
      convInfo.effectiveFilterHeight,
      convInfo.effectiveFilterWidth,
      convInfo.padInfo.top,
      convInfo.padInfo.left,
      convInfo.filterHeight,
      convInfo.filterWidth,
  );
  return dx;
}

export const avgPoolGradConfig: KernelConfig = {
  kernelName: AvgPoolGrad,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: avgPoolGrad as unknown as KernelFunc
};
