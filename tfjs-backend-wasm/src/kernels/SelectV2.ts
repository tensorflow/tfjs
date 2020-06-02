/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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

import {NamedTensorInfoMap, registerKernel, SelectV2, SelectV2Inputs} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmSelect: (
    conditionId: number, tId: number, eId: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmSelect = backend.wasm.cwrap(SelectV2, null, [
    'number',  // conditionId
    'number',  // tId
    'number',  // eId
    'number',  // outId
  ]);
}

function select(args: {inputs: NamedTensorInfoMap, backend: BackendWasm}) {
  const {inputs, backend} = args;
  const {condition, t, e} = inputs as {} as SelectV2Inputs;

  const conditionId = backend.dataIdMap.get(condition.dataId).id;
  const tId = backend.dataIdMap.get(t.dataId).id;
  const eId = backend.dataIdMap.get(e.dataId).id;
  const out = backend.makeOutput(t.shape, t.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;

  wasmSelect(conditionId, tId, eId, outId);
  return out;
}

registerKernel({
  kernelName: SelectV2,
  backendName: 'wasm',
  kernelFunc: select,
  setupFunc: setup
});
