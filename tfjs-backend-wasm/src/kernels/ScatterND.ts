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

import {NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface ScatterNDInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface ScatterNDAttrs extends NamedAttrMap {
  newWidth: number;
  newHeight: number;
  alignCorners: boolean;
}

let wasmScatterND: (indicesId: number, updatesId: number, outId: number) =>
    void;

function setup(backend: BackendWasm): void {
  wasmScatterND = backend.wasm.cwrap('ScatterND', null /*void*/, [
    'number',  // indicesId
    'number',  // updatesId
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

  const out = backend.makeOutput(shape as number[], updates.dtype);
  if (util.sizeFromShape(shape as number[]) === 0) {
    return out;
  }

  const indicesData = backend.dataIdMap.get(indices.dataId);
  const indicesId = indicesData.id;

  const updatesData = backend.dataIdMap.get(updates.dataId);
  const updatesId = updatesData.id;

  const outId = backend.dataIdMap.get(out.dataId).id;
  wasmScatterND(indicesId, updatesId, outId);

  return out;
}

registerKernel({
  kernelName: 'ScatterND',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: scatterND
});
