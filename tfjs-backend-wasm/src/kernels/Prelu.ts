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

import {KernelConfig, KernelFunc, Prelu, PreluInputs} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmPrelu: (xId: number, weightsId: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmPrelu = backend.wasm.cwrap(Prelu, null /* void */, [
    'number',  // x_id
    'number',  // weights_id
    'number'   // out_id
  ]);
}

function prelu(args: {inputs: PreluInputs, backend: BackendWasm}) {
  const {inputs, backend} = args;
  const {x, alpha} = inputs;
  const xId = backend.dataIdMap.get(x.dataId).id;
  const weightsId = backend.dataIdMap.get(alpha.dataId).id;

  const out = backend.makeOutput(x.shape, 'float32');
  const outId = backend.dataIdMap.get(out.dataId).id;
  wasmPrelu(xId, weightsId, outId);
  return out;
}

export const preluConfig: KernelConfig = {
  kernelName: Prelu,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: prelu as {} as KernelFunc
};
