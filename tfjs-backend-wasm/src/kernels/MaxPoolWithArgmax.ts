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

import {backend_util, KernelConfig, KernelFunc, MaxPoolWithArgmax, MaxPoolWithArgmaxAttrs, MaxPoolWithArgmaxInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {CppDType} from './types';

let wasmMaxPoolWithArgmax: (
    xId: number, pooledId: number, indexesId: number, dtype: number,
    includeBatchIndex: boolean, batchSize: number, channelSize: number,
    inHeight: number, inWidth: number, outHeight: number, outWidth: number,
    strideHeight: number, strideWidth: number, dilationHeight: number,
    dilationWidth: number, effectiveFilterHeight: number,
    effectiveFilterWidth: number, padTop: number, padLeft: number) => void;

function setup(backend: BackendWasm) {
  wasmMaxPoolWithArgmax = backend.wasm.cwrap('MaxPoolWithArgmax', null, [
    'number',   // xId
    'number',   // pooledId
    'number',   // indexesId
    'number',   // dtype
    'boolean',  // includeBatchIndex
    'number',   // batchSize
    'number',   // channelSize
    'number',   // inHeight
    'number',   // inWidth
    'number',   // outHeight
    'number',   // outWidth
    'number',   // strideHeight
    'number',   // strideWidth
    'number',   // dilationHeight
    'number',   // dilationWidth
    'number',   // effectiveFilterHeight
    'number',   // effectiveFilterWidth
    'number',   // padTop
    'number',   // padLeft
  ]);
}

export function maxPoolWithArgmax(args: {
  inputs: MaxPoolWithArgmaxInputs,
  attrs: MaxPoolWithArgmaxAttrs,
  backend: BackendWasm,
}): TensorInfo[] {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {filterSize, strides, pad, includeBatchInIndex} = attrs;

  util.assert(
      x.shape.length === 4,
      () => `Error in maxPool: input must be rank 4 but got rank ${
          x.shape.length}.`);
  const dilations: [number, number] = [1, 1];
  util.assert(
      backend_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in maxPool: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);

  const convInfo = backend_util.computePool2DInfo(
      x.shape as [number, number, number, number], filterSize, strides, [1, 1],
      pad);

  const pooled = backend.makeOutput(convInfo.outShape, x.dtype);
  const indexes = backend.makeOutput(convInfo.outShape, 'int32');

  wasmMaxPoolWithArgmax(
      backend.dataIdMap.get(x.dataId).id,
      backend.dataIdMap.get(pooled.dataId).id,
      backend.dataIdMap.get(indexes.dataId).id,
      CppDType[x.dtype],
      includeBatchInIndex,
      convInfo.batchSize,
      convInfo.inChannels,
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
  return [pooled, indexes];
}

export const maxPoolWithArgmaxConfig: KernelConfig = {
  kernelName: MaxPoolWithArgmax,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: maxPoolWithArgmax as unknown as KernelFunc
};
