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

import {FusedBatchNorm, FusedBatchNormAttrs, FusedBatchNormInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmBatchNorm: (
    xId: number, meanId: number, varianceId: number, offsetId: number,
    scaleId: number, varianceEpsilon: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmBatchNorm = backend.wasm.cwrap(
      FusedBatchNorm, null /* void */,
      ['number', 'number', 'number', 'number', 'number', 'number', 'number']);
}

function fusedBatchNorm(args: {
  backend: BackendWasm,
  inputs: FusedBatchNormInputs,
  attrs: FusedBatchNormAttrs
}): TensorInfo {
  const {backend, inputs, attrs} = args;
  const {varianceEpsilon} = attrs;
  const {x, mean, variance, offset, scale} = inputs;
  const xId = backend.dataIdMap.get(x.dataId).id;
  const meanId = backend.dataIdMap.get(mean.dataId).id;
  const varianceId = backend.dataIdMap.get(variance.dataId).id;
  const offsetId = offset != null ? backend.dataIdMap.get(offset.dataId).id : 0;
  const scaleId = scale != null ? backend.dataIdMap.get(scale.dataId).id : 0;

  const out = backend.makeOutput(x.shape, x.dtype);
  // Short-circuit zero-sized tensors.
  if (util.sizeFromShape(x.shape) === 0) {
    return out;
  }

  const outId = backend.dataIdMap.get(out.dataId).id;

  wasmBatchNorm(
      xId, meanId, varianceId, offsetId, scaleId, varianceEpsilon, outId);
  return out;
}

export const fusedBatchNormConfig: KernelConfig = {
  kernelName: FusedBatchNorm,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: fusedBatchNorm as {} as KernelFunc
};
