/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {NamedAttrMap, NamedTensorInfoMap, registerKernel, scatter_nd_util, Tensor, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface ScatterNDInputs extends NamedTensorInfoMap {
  indices: TensorInfo;
  updates: TensorInfo;
}

interface ScatterNDAttrs extends NamedAttrMap {
  shape: number[];
}

let wasmScatterND: (
    indicesId: number, updatesId: number, sliceRank: number, numUpdates: number,
    sliceSize: number, strides: Uint8Array, shape: Uint8Array,
    outputSize: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmScatterND = backend.wasm.cwrap('ScatterND', null /*void*/, [
    'number',  // indicesId
    'number',  // updatesId
    'number',  // sliceRank
    'number',  // numUpdates
    'number',  // sliceSize
    'array',   // strides
    'array',   // shape
    'number',  // outputSize
    'number'   // outId
  ]);
}

function scatterND(
    args:
        {backend: BackendWasm, inputs: ScatterNDInputs, attrs: ScatterNDAttrs}):
    TensorInfo {
  const {backend, inputs, attrs} = args;
  const {indices, updates} = inputs;
  const {shape} = attrs;

  const out = backend.makeOutput(shape, updates.dtype);
  if (util.sizeFromShape(shape) === 0) {
    return out;
  }

  const {sliceRank, numUpdates, sliceSize, strides, outputSize} =
      scatter_nd_util.calculateShapes(
          updates as Tensor, indices as Tensor, shape);

  const indicesData = backend.dataIdMap.get(indices.dataId);
  const indicesId = indicesData.id;

  const updatesData = backend.dataIdMap.get(updates.dataId);
  const updatesId = updatesData.id;

  const stridesBytes = new Uint8Array(new Int32Array(strides).buffer);
  const shapeBytes = new Uint8Array(new Int32Array(shape).buffer);

  const outId = backend.dataIdMap.get(out.dataId).id;
  wasmScatterND(
      indicesId, updatesId, sliceRank, numUpdates, sliceSize, stridesBytes,
      shapeBytes, outputSize, outId);

  return out;
}

registerKernel({
  kernelName: 'ScatterND',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: scatterND
});
