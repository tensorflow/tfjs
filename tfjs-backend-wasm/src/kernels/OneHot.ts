/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {KernelConfig, KernelFunc, OneHot, OneHotAttrs, OneHotInputs} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmOneHot: (
    indicesId: number, depth: number, onValue: number, offValue: number,
    outId: number) => void;

function setup(backend: BackendWasm) {
  wasmOneHot = backend.wasm.cwrap(OneHot, null /* void */, [
    'number',  // indices_id
    'number',  // depth,
    'number',  // onValue
    'number',  // offValue
    'number'   // out_id
  ]);
}

function oneHot(
    args: {inputs: OneHotInputs, attrs: OneHotAttrs, backend: BackendWasm}) {
  const {inputs, backend, attrs} = args;
  const {indices} = inputs;
  const {depth, onValue, offValue} = attrs;

  const out = backend.makeOutput([...indices.shape, depth], 'int32');
  const outId = backend.dataIdMap.get(out.dataId).id;

  const indicesData = backend.dataIdMap.get(indices.dataId);
  const indicesId = indicesData.id;

  wasmOneHot(indicesId, depth, onValue, offValue, outId);

  return out;
}

export const oneHotConfig: KernelConfig = {
  kernelName: OneHot,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: oneHot as {} as KernelFunc,
};
