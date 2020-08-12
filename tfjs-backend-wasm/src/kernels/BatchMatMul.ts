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

import {BatchMatMul, BatchMatMulAttrs, BatchMatMulInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmBatchMatMul: (
    aId: number, aShape: Uint8Array, aShapeSize: number, bId: number,
    bShape: Uint8Array, bShapeSize: number, transposeA: boolean,
    transposeB: boolean, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmBatchMatMul = backend.wasm.cwrap(BatchMatMul, null /* void */, [
    'number',  // a_id
    'array',   // a_shape
    'number',  // a_shape.length
    'number',  // b_id
    'array',   // b_shape
    'number',  // b_shape.length
    'number',  // transpose_a
    'number',  // transpose_b
    'number'   // out_id
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

  const leftDim = transposeA ? a.shape[2] : a.shape[1];
  const rightDim = transposeB ? b.shape[1] : b.shape[2];
  const batchDim = a.shape[0];

  const out = backend.makeOutput([batchDim, leftDim, rightDim], a.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;

  const aShapeBytes = new Uint8Array(new Int32Array(a.shape).buffer);
  const bShapeBytes = new Uint8Array(new Int32Array(b.shape).buffer);

  wasmBatchMatMul(
      aId, aShapeBytes, a.shape.length, bId, bShapeBytes, b.shape.length,
      transposeA, transposeB, outId);

  return out;
}

export const batchMatMulConfig: KernelConfig = {
  kernelName: BatchMatMul,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: batchMatMul as {} as KernelFunc
};
