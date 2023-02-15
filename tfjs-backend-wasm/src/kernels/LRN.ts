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

import {KernelConfig, KernelFunc, LRN, LRNAttrs, LRNInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmLRN: (
    xId: number, outId: number, channels: number, depthRadius: number,
    bias: number, alpha: number, beta: number) => void;

function setup(backend: BackendWasm) {
  wasmLRN = backend.wasm.cwrap(LRN, null, [
    'number',  // xId
    'number',  // outId
    'number',  // channels
    'number',  // depthRadius
    'number',  // bias
    'number',  // alpha
    'number',  // beta
  ]);
}

export function lrn(args: {
  inputs: LRNInputs,
  attrs: LRNAttrs,
  backend: BackendWasm,
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {depthRadius, bias, alpha, beta} = attrs;

  if (x.dtype !== 'float32') {
    throw new Error('LRN error: x must have dtype float32');
  }

  const out = backend.makeOutput(x.shape, x.dtype);

  wasmLRN(
      backend.dataIdMap.get(x.dataId).id,
      backend.dataIdMap.get(out.dataId).id,
      /*channels=*/x.shape[3],
      depthRadius,
      bias,
      alpha,
      beta,
  );
  return out;
}

export const lrnConfig: KernelConfig = {
  kernelName: LRN,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: lrn as unknown as KernelFunc
};
