/**
 * @license
 * Copyright 2023 Google LLC.
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

import {Diag, DiagInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {CppDType} from './types';

let wasmDiag: (xId: number, xDType: CppDType, xSize: number, outId: number) =>
    void;

function setup(backend: BackendWasm) {
  wasmDiag = backend.wasm.cwrap('Diag', null, [
    'number',  // xId
    'number',  // xDType,
    'number',  // xSize,
    'number',  // outId
  ]);
}

export function diag(args: {inputs: DiagInputs, backend: BackendWasm}):
    TensorInfo {
  const {inputs, backend} = args;
  const {x} = inputs;

  const xSize = util.sizeFromShape(x.shape);
  const out = backend.makeOutput([...x.shape, ...x.shape], x.dtype);

  wasmDiag(
      backend.dataIdMap.get(x.dataId).id, CppDType[x.dtype], xSize,
      backend.dataIdMap.get(out.dataId).id);
  return out;
}

export const diagConfig: KernelConfig = {
  kernelName: Diag,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: diag as unknown as KernelFunc
};
