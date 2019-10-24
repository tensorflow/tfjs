/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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

import {NamedTensorInfoMap, registerKernel, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface BatchNormInputs extends NamedTensorInfoMap {
  x: TensorInfo;
  mean: TensorInfo;
  variance: TensorInfo;
  offset: TensorInfo;
  scale: TensorInfo;
  varianceEpsilon: TensorInfo;
}

let wasmBatchNorm: (
    xId: number, meanId: number, varianceId: number, offsetId: number,
    scaleId: number, varianceEpsilonId: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmBatchNorm = backend.wasm.cwrap(
      'BatchNormalization', null /* void */,
      ['number', 'number', 'number', 'number', 'number', 'number', 'number']);
}

function batchNormalization(
    args: {backend: BackendWasm, inputs: BatchNormInputs}): TensorInfo {
  const {backend, inputs} = args;
  const {x, mean, variance, offset, scale, varianceEpsilon} = inputs;
  const xId = backend.dataIdMap.get(x.dataId).id;
  const meanId = backend.dataIdMap.get(mean.dataId).id;
  const varianceId = backend.dataIdMap.get(variance.dataId).id;
  const offsetId = backend.dataIdMap.get(offset.dataId).id;
  const scaleId = backend.dataIdMap.get(scale.dataId).id;
  const varianceEpsilonId = backend.dataIdMap.get(varianceEpsilon.dataId).id;

  const out = backend.makeOutput(x.shape, x.dtype);
  // Short-circuit zero-sized tensors.
  if (util.sizeFromShape(x.shape) === 0) {
    return out;
  }

  const outId = backend.dataIdMap.get(out.dataId).id;

  wasmBatchNorm(
      xId, meanId, varianceId, offsetId, scaleId, varianceEpsilonId, outId);
  return out;
}

registerKernel({
  kernelName: 'BatchNormalization',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: batchNormalization
});
