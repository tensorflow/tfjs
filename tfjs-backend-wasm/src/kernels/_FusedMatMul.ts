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

import {_FusedMatMul, _FusedMatMulAttrs, _FusedMatMulInputs, broadcast_util, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {FusableActivation} from './types';

let wasmFusedMatMul:
    (aId: number, aShape: Uint8Array, aShapeSize: number, bId: number,
     bShape: Uint8Array, bShapeSize: number, transposeA: boolean,
     transposeB: boolean, activation: number, biasId: number,
     preluActivationWeightsId: number, leakyreluAlpha: number, outId: number) =>
        void;

function setup(backend: BackendWasm) {
  wasmFusedMatMul = backend.wasm.cwrap(_FusedMatMul, null /* void */, [
    'number',  // a_id
    'array',   // a_shape
    'number',  // a_shape.length
    'number',  // b_id
    'array',   // b_shape
    'number',  // b_shape.length
    'number',  // transpose_a
    'number',  // transpose_b
    'number',  // activation
    'number',  // biasId
    'number',  // preluActivationWeightsId
    'number',  // leakyreluAlpha
    'number'   // out_id
  ]);
}

function fusedBatchMatMul(args: {
  inputs: _FusedMatMulInputs,
  backend: BackendWasm,
  attrs: _FusedMatMulAttrs
}) {
  const {inputs, backend, attrs} = args;
  const {a, b, bias, preluActivationWeights} = inputs;

  if (a.dtype !== 'float32' || b.dtype !== 'float32') {
    throw new Error(
        `_FusedMatMul for non non-float32 tensors not yet supported.`);
  }

  const {transposeA, transposeB, activation, leakyreluAlpha} = attrs;
  const aId = backend.dataIdMap.get(a.dataId).id;
  const bId = backend.dataIdMap.get(b.dataId).id;

  let biasId = 0;
  if (bias != null) {
    const biasData = backend.dataIdMap.get(bias.dataId);
    if (biasData.shape.length !== 1) {
      throw new Error(
          `_FusedMatMul only supports rank-1 bias but got ` +
          `rank ${biasData.shape.length}.`);
    }
    biasId = biasData.id;
  }
  const preluActivationWeightsId = preluActivationWeights == null ?
      0 :
      backend.dataIdMap.get(preluActivationWeights.dataId).id;
  const fusedActivation =
      FusableActivation[activation as {} as keyof typeof FusableActivation];
  if (fusedActivation == null) {
    throw new Error(
        `${activation} activation not yet supported for FusedConv2D ` +
        `in the wasm backend.`);
  }

  const leftDim = transposeA ? a.shape[2] : a.shape[1];
  const rightDim = transposeB ? b.shape[1] : b.shape[2];
  const batchDims = broadcast_util.assertAndGetBroadcastShape(
      a.shape.slice(0, -2), b.shape.slice(0, -2));

  const out = backend.makeOutput([...batchDims, leftDim, rightDim], a.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;

  const aShapeBytes = new Uint8Array(new Int32Array(a.shape).buffer);
  const bShapeBytes = new Uint8Array(new Int32Array(b.shape).buffer);

  wasmFusedMatMul(
      aId, aShapeBytes, a.shape.length, bId, bShapeBytes, b.shape.length,
      transposeA, transposeB, fusedActivation, biasId, preluActivationWeightsId,
      leakyreluAlpha || 0, outId);

  return out;
}

export const _fusedMatMulConfig: KernelConfig = {
  kernelName: _FusedMatMul,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: fusedBatchMatMul as {} as KernelFunc
};
