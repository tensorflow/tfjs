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

import {KernelConfig, KernelFunc, scatter_util, ScatterNd, ScatterNdAttrs, ScatterNdInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {CppDType} from './types';

let wasmScatterNd: (
    indicesId: number, updatesId: number, dtype: CppDType, sliceRank: number,
    numUpdates: number, sliceSize: number, strides: Uint8Array,
    outputSize: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmScatterNd = backend.wasm.cwrap(ScatterNd, null /*void*/, [
    'number',  // indicesId
    'number',  // updatesId
    'number',  // dtype
    'number',  // sliceRank
    'number',  // numUpdates
    'number',  // sliceSize
    'array',   // strides
    'number',  // outputSize
    'number'   // outId
  ]);
}

function scatterNd(
    args:
        {backend: BackendWasm, inputs: ScatterNdInputs, attrs: ScatterNdAttrs}):
    TensorInfo {
  const {backend, inputs, attrs} = args;
  const {indices, updates} = inputs;
  const {shape} = attrs;

  const out = backend.makeOutput(shape, updates.dtype);
  if (util.sizeFromShape(shape) === 0) {
    return out;
  }

  const {sliceRank, numUpdates, sliceSize, strides, outputSize} =
      scatter_util.calculateShapes(updates, indices, shape);

  const indicesData = backend.dataIdMap.get(indices.dataId);
  const indicesId = indicesData.id;

  const updatesData = backend.dataIdMap.get(updates.dataId);
  const updatesId = updatesData.id;

  const stridesBytes = new Uint8Array(new Int32Array(strides).buffer);

  const outId = backend.dataIdMap.get(out.dataId).id;
  wasmScatterNd(
      indicesId, updatesId, CppDType[updates.dtype], sliceRank, numUpdates,
      sliceSize, stridesBytes, outputSize, outId);

  return out;
}

export const scatterNdConfig: KernelConfig = {
  kernelName: ScatterNd,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: scatterNd as {} as KernelFunc
};
