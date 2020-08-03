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

import {AddN, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {CppDType} from './types';

let wasmFunc:
    (inputIds: Uint8Array, inputIdsLen: number, dtype: number, outId: number) =>
        void;

function setupFunc(backend: BackendWasm): void {
  wasmFunc = backend.wasm.cwrap(AddN, null /* void */, [
    'array',   // input_ids
    'number',  // input_ids.length
    'number',  // dtype
    'number',  // out_id
  ]);
}

function addn(args: {inputs: TensorInfo[], backend: BackendWasm}) {
  const {inputs, backend} = args;
  const out = backend.makeOutput(inputs[0].shape, inputs[0].dtype);

  // Short-circuit zero-sized tensors.
  if (util.sizeFromShape(out.shape) === 0) {
    return out;
  }

  const inputIds = inputs.map(x => backend.dataIdMap.get(x.dataId).id);
  const inputIdsBytes = new Uint8Array(new Int32Array(inputIds).buffer);
  const outId = backend.dataIdMap.get(out.dataId).id;
  wasmFunc(inputIdsBytes, inputIds.length, CppDType[out.dtype], outId);

  return out;
}

export const addNConfig: KernelConfig = {
  kernelName: AddN,
  backendName: 'wasm',
  setupFunc,
  kernelFunc: addn as {} as KernelFunc,
};
