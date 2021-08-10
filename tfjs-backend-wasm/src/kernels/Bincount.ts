/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import {Bincount, BincountAttrs, BincountInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmFunc: (xId: number, weightsId: number, size: number, outId: number) =>
    void;

function setup(backend: BackendWasm) {
  wasmFunc = backend.wasm.cwrap(Bincount, null /* void */, [
    'number',  // xId
    'number',  // weightsId
    'number',  // size
    'number'   // outId
  ]);
}

function bincount(
    args:
        {backend: BackendWasm, inputs: BincountInputs, attrs: BincountAttrs}) {
  const {backend, inputs, attrs} = args;
  const {x, weights} = inputs;
  const {size} = attrs;

  const xId = backend.dataIdMap.get(x.dataId).id;
  const weightsId = backend.dataIdMap.get(weights.dataId).id;

  const out = backend.makeOutput([size], weights.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;

  const outVals = backend.typedArrayFromHeap(out);
  outVals.fill(0);

  wasmFunc(xId, weightsId, size, outId);

  return out;
}

export const bincountConfig: KernelConfig = {
  kernelName: Bincount,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: bincount as {} as KernelFunc
};
