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

import {NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface BatchMatMulInputs extends NamedTensorInfoMap {
  a: TensorInfo;
  b: TensorInfo;
}

interface BatchMatMulAttrs extends NamedAttrMap {
  transposeA: boolean;
  transposeB: boolean;
}

let wasmBatchMatMul: (
    aId: number, bId: number, sharedDim: number, leftDim: number,
    rightDim: number, batchDim: number, aBatch: number, aOuterStep: number,
    aInnerStep: number, bBatch: number, bOuterStep: number, bInnerStep: number,
    outId: number) => void;

function setup(backend: BackendWasm) {
  wasmBatchMatMul = backend.wasm.cwrap('BatchMatMul', null /* void */, [
    'number', 'number', 'number', 'number', 'number', 'number', 'number',
    'number', 'number', 'number', 'number', 'number', 'number'
  ]);
}

function batchMatMul(args: {
  inputs: BatchMatMulInputs,
  backend: BackendWasm,
  attrs: BatchMatMulAttrs
}) {
  const {inputs, backend, attrs} = args;
  const {a, b} = inputs;

  if (a.dtype !== 'float32' || b.dtype !== 'float32') {
    throw new Error(
        `BatchMatMul for non non-float32 tensors not yet supported.`);
  }

  const {transposeA, transposeB} = attrs;
  const aId = backend.dataIdMap.get(a.dataId).id;
  const bId = backend.dataIdMap.get(b.dataId).id;

  const sharedDim = transposeA ? a.shape[1] : a.shape[2];
  const leftDim = transposeA ? a.shape[2] : a.shape[1];
  const rightDim = transposeB ? b.shape[1] : b.shape[2];
  const batchDim = a.shape[0];

  const aStrides = util.computeStrides(a.shape);
  const bStrides = util.computeStrides(b.shape);
  const [aBatch, aOuterStep, aInnerStep] = transposeA ?
      [aStrides[0], 1, aStrides[1]] :
      [aStrides[0], aStrides[1], 1];
  const [bInnerStep, bOuterStep, bBatch] = transposeB ?
      [1, bStrides[1], bStrides[0]] :
      [bStrides[1], 1, bStrides[0]];

  const out = backend.makeOutput([batchDim, leftDim, rightDim], a.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;

  wasmBatchMatMul(
      aId, bId, sharedDim, leftDim, rightDim, batchDim, aBatch, aOuterStep,
      aInnerStep, bBatch, bOuterStep, bInnerStep, outId);
  return out;
}

registerKernel({
  kernelName: 'BatchMatMul',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: batchMatMul
});
