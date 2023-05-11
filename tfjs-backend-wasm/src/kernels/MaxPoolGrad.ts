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

import {backend_util, KernelConfig, KernelFunc, MaxPoolGrad, MaxPoolGradAttrs, MaxPoolGradInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmMaxPoolGrad: (
    xId: number, dyId: number, dxId: number, batchSize: number,
    channelSize: number, inHeight: number, inWidth: number, outHeight: number,
    outWidth: number, strideHeight: number, strideWidth: number,
    dilationHeight: number, dilationWidth: number,
    effectiveFilterHeight: number, effectiveFilterWidth: number, padTop: number,
    padLeft: number) => void;

function setup(backend: BackendWasm) {
  wasmMaxPoolGrad = backend.wasm.cwrap('MaxPoolGrad', null, [
    'number',  // xId
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
  ]);
}

export function maxPoolGrad(args: {
  inputs: MaxPoolGradInputs,
  attrs: MaxPoolGradAttrs,
  backend: BackendWasm,
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, input} = inputs;
  const {filterSize, strides, pad, dimRoundingMode} = attrs;

  const convInfo = backend_util.computePool2DInfo(
      input.shape as [number, number, number, number], filterSize, strides,
      /*dilations=*/1, pad, dimRoundingMode);
  const dx = backend.makeOutput(input.shape, input.dtype);

  wasmMaxPoolGrad(
      backend.dataIdMap.get(input.dataId).id,
      backend.dataIdMap.get(dy.dataId).id,
      backend.dataIdMap.get(dx.dataId).id,
      convInfo.batchSize,
      // Since Pool ops (MaxPool and MaxPool) support 2D filter only, in
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
  );
  return dx;
}

export const maxPoolGradConfig: KernelConfig = {
  kernelName: MaxPoolGrad,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: maxPoolGrad as unknown as KernelFunc
};
