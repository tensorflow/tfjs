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

import {EluGrad, EluGradInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmEluGrad: (yId: number, dyId: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmEluGrad = backend.wasm.cwrap(EluGrad, null, [
    'number',  // yId
    'number',  // dyId,
    'number',  // outId
  ]);
}

export function eluGrad(args: {inputs: EluGradInputs, backend: BackendWasm}):
    TensorInfo {
  const {inputs, backend} = args;
  const {dy, y} = inputs;

  const out = backend.makeOutput(y.shape, 'float32');
  const tensorId = (x: TensorInfo) => {
    return backend.dataIdMap.get(x.dataId).id!;
  };
  wasmEluGrad(tensorId(y), tensorId(dy), tensorId(out));
  return out;
}

export const eluGradConfig: KernelConfig = {
  kernelName: EluGrad,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: eluGrad as unknown as KernelFunc,
};
