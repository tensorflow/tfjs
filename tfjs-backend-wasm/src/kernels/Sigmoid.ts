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

import {KernelConfig, KernelFunc, Sigmoid, SigmoidInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmFunc: (xId: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmFunc = backend.wasm.cwrap(Sigmoid, null /* void */, ['number', 'number']);
}

function sigmoid(args: {backend: BackendWasm, inputs: SigmoidInputs}):
    TensorInfo {
  const {backend, inputs: {x}} = args;
  const xId = backend.dataIdMap.get(x.dataId).id;
  const out = backend.makeOutput(x.shape, x.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;

  // Short-circuit zero-sized tensors.
  if (util.sizeFromShape(out.shape) === 0) {
    return out;
  }

  wasmFunc(xId, outId);
  return out;
}

export const sigmoidConfig: KernelConfig = {
  kernelName: 'Sigmoid',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: sigmoid as {} as KernelFunc
};
