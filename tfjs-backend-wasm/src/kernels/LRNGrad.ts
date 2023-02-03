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

import {KernelConfig, KernelFunc, LRNGrad, LRNGradAttrs, LRNGradInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmLRNGrad: (
    xId: number, yId: number, dyId: number, dxId: number, channels: number,
    depthRadius: number, bias: number, alpha: number, beta: number) => void;

function setup(backend: BackendWasm) {
  wasmLRNGrad = backend.wasm.cwrap(LRNGrad, null, [
    'number',  // xId
    'number',  // yId
    'number',  // dyId
    'number',  // dxId
    'number',  // channels
    'number',  // depthRadius
    'number',  // bias
    'number',  // alpha
    'number',  // beta
  ]);
}

export function lrnGrad(args: {
  inputs: LRNGradInputs,
  attrs: LRNGradAttrs,
  backend: BackendWasm,
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x, y, dy} = inputs;
  const {depthRadius, bias, alpha, beta} = attrs;

  if (x.dtype !== 'float32' || y.dtype !== 'float32' ||
      dy.dtype !== 'float32') {
    throw new Error('LRNGrad error: x, y, and dy must have dtype float32');
  }

  const dx = backend.makeOutput(x.shape, x.dtype);

  wasmLRNGrad(
      backend.dataIdMap.get(x.dataId).id,
      backend.dataIdMap.get(y.dataId).id,
      backend.dataIdMap.get(dy.dataId).id,
      backend.dataIdMap.get(dx.dataId).id,
      /*channels=*/dy.shape[3],
      depthRadius,
      bias,
      alpha,
      beta,
  );
  return dx;
}

export const lrnGradConfig: KernelConfig = {
  kernelName: LRNGrad,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: lrnGrad as unknown as KernelFunc
};
