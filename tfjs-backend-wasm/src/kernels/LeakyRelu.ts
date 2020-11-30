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

import {KernelConfig, KernelFunc, LeakyRelu, LeakyReluAttrs, LeakyReluInputs, util} from '@tensorflow/tfjs-core';
import {TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmFunc: (xId: number, leakyreluAlpha: number, outId: number) => void;

function setupFunc(backend: BackendWasm): void {
  wasmFunc = backend.wasm.cwrap(LeakyRelu, null /* void */, [
    'number',  // x_id,
    'number',  // leakyrelu_alpha
    'number'   // out_id
  ]);
}

export function leakyRelu(
    args:
        {inputs: LeakyReluInputs, attrs: LeakyReluAttrs, backend: BackendWasm}):
    TensorInfo {
  const {inputs: {x}, attrs: {alpha}, backend} = args;

  const xId = backend.dataIdMap.get(x.dataId).id;
  const out = backend.makeOutput(x.shape, x.dtype);

  if (util.sizeFromShape(x.shape) !== 0) {
    const outId = backend.dataIdMap.get(out.dataId).id;
    wasmFunc(xId, alpha, outId);
  }

  return out;
}

export const leakyReluConfig: KernelConfig = {
  kernelName: LeakyRelu,
  backendName: 'wasm',
  setupFunc,
  kernelFunc: leakyRelu as {} as KernelFunc,
};
