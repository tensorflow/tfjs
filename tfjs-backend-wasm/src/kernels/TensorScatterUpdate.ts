/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {KernelConfig, KernelFunc, scatter_util, TensorInfo, TensorScatterUpdate, TensorScatterUpdateAttrs, TensorScatterUpdateInputs, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {CppDType} from './types';

let wasmTensorScatterUpdate: (
    indicesId: number, updatesId: number, dtype: CppDType, sliceRank: number,
    numUpdates: number, sliceSize: number, strides: Uint8Array,
    outputSize: number, outId: number, tensorId: number) => void;

function setup(backend: BackendWasm): void {
  wasmTensorScatterUpdate =
      backend.wasm.cwrap(TensorScatterUpdate, null /*void*/, [
        'number',  // indicesId
        'number',  // updatesId
        'number',  // dtype
        'number',  // sliceRank
        'number',  // numUpdates
        'number',  // sliceSize
        'array',   // strides
        'number',  // outputSize
        'number',  // outId
        'number',  // tensorId
      ]);
}

function tensorScatterUpdate(args: {
  backend: BackendWasm,
  inputs: TensorScatterUpdateInputs,
  attrs: TensorScatterUpdateAttrs
}): TensorInfo {
  const {backend, inputs, attrs} = args;
  const {tensor, indices, updates} = inputs;
  const {} = attrs;

  const out = backend.makeOutput(tensor.shape, tensor.dtype);
  if (util.sizeFromShape(tensor.shape) === 0) {
    return out;
  }

  const {sliceRank, numUpdates, sliceSize, strides, outputSize} =
      scatter_util.calculateShapes(updates, indices, tensor.shape);

  const indicesData = backend.dataIdMap.get(indices.dataId);
  const indicesId = indicesData.id;

  const updatesData = backend.dataIdMap.get(updates.dataId);
  const updatesId = updatesData.id;

  const tensorData = backend.dataIdMap.get(tensor.dataId);
  const tensorId = tensorData.id;

  const stridesBytes = new Uint8Array(new Int32Array(strides).buffer);

  const outId = backend.dataIdMap.get(out.dataId).id;
  wasmTensorScatterUpdate(
      indicesId, updatesId, CppDType[updates.dtype], sliceRank, numUpdates,
      sliceSize, stridesBytes, outputSize, outId, tensorId);

  return out;
}

export const tensorScatterUpdateConfig: KernelConfig = {
  kernelName: TensorScatterUpdate,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: tensorScatterUpdate as unknown as KernelFunc
};
