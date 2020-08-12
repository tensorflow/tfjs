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

import {KernelConfig, KernelFunc, Softmax, SoftmaxAttrs, SoftmaxInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmFunc: (xId: number, outId: number, channels: number, batch: number) =>
    void;

function setup(backend: BackendWasm): void {
  wasmFunc = backend.wasm.cwrap(Softmax, null /* void */, [
    'number',  // xId
    'number',  // outId
    'number',  // channels
    'number'   // batch
  ]);
}

function softmax(
    args: {backend: BackendWasm, inputs: SoftmaxInputs, attrs: SoftmaxAttrs}):
    TensorInfo {
  const {backend, inputs: {logits}, attrs: {dim}} = args;
  const xId = backend.dataIdMap.get(logits.dataId).id;
  const out = backend.makeOutput(logits.shape, logits.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;

  const channels = logits.shape[dim];
  const batch = util.sizeFromShape(logits.shape) / channels;

  // Short-circuit zero-sized tensors.
  if (util.sizeFromShape(out.shape) === 0) {
    return out;
  }

  wasmFunc(xId, outId, channels, batch);
  return out;
}

export const softmaxConfig: KernelConfig = {
  kernelName: Softmax,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: softmax as {} as KernelFunc
};
