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

import {gather_util, GatherNd, GatherNdInputs, KernelConfig, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {CppDType} from './types';

let wasmGatherNd: (
    xId: number, dtype: CppDType, indicesId: number, numSlices: number,
    sliceRank: number, sliceSize: number, strides: Uint8Array, outId: number) =>
    void;

function setup(backend: BackendWasm): void {
  wasmGatherNd = backend.wasm.cwrap(GatherNd, null /*void*/, [
    'number',  // xId
    'number',  // dtype
    'number',  // indicesId
    'number',  // numSlices
    'number',  // sliceRank
    'number',  // sliceSize
    'array',   // strides
    'number'   // outId
  ]);
}

function gatherNd(args: {backend: BackendWasm, inputs: GatherNdInputs}):
    TensorInfo {
  const {backend, inputs} = args;
  const {params, indices} = inputs;

  const [resultShape, numSlices, sliceSize, strides] =
      gather_util.prepareAndValidate(params, indices);

  const out = backend.makeOutput(resultShape, params.dtype);
  if (numSlices === 0) {
    return out;
  }

  const indicesShape = indices.shape;
  const sliceRank = indicesShape[indicesShape.length - 1];

  const xData = backend.dataIdMap.get(params.dataId);
  const xId = xData.id;
  const indicesData = backend.dataIdMap.get(indices.dataId);
  const indicesId = indicesData.id;

  const stridesBytes = new Uint8Array(new Int32Array(strides).buffer);

  const outId = backend.dataIdMap.get(out.dataId).id;
  wasmGatherNd(
      xId, CppDType[params.dtype], indicesId, numSlices, sliceRank, sliceSize,
      stridesBytes, outId);

  return out;
}

export const gatherNdConfig: KernelConfig = {
  kernelName: GatherNd,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: gatherNd
};
