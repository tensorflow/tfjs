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

import {BatchMatMul, BatchMatMulAttrs, BatchMatMulInputs, KernelConfig, KernelFunc, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {reshape} from './Reshape';

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
  const {transposeA, transposeB} = attrs;

  if (a.dtype !== 'float32' || b.dtype !== 'float32') {
    throw new Error(
        `BatchMatMul for non non-float32 tensors not yet supported.`);
  }

  const aRank = a.shape.length;
  const bRank = b.shape.length;

  const innerShapeA = transposeA ? a.shape[aRank - 2] : a.shape[aRank - 1];
  const innerShapeB = transposeB ? b.shape[bRank - 1] : b.shape[bRank - 2];

  const outerShapeA = transposeA ? a.shape[aRank - 1] : a.shape[aRank - 2];
  const outerShapeB = transposeB ? b.shape[bRank - 2] : b.shape[bRank - 1];

  const outerDimsA = a.shape.slice(0, -2);
  const outerDimsB = b.shape.slice(0, -2);

  const batchDimA = util.sizeFromShape(outerDimsA);
  const batchDimB = util.sizeFromShape(outerDimsB);

  const batchDimsCompatible =
      batchDimA === batchDimB || batchDimA === 1 || batchDimB === 1;

  util.assert(
      aRank >= 2 && bRank >= 2 && batchDimsCompatible,
      () => `Error in matMul: the input batch dimensions must either be the ` +
          `same or at least one input batch dimension must be 1. Got input ` +
          `batch dimensions of (${outerDimsA}) and (${outerDimsB}).`);

  const outShapeOuterDims =
      batchDimA > batchDimB ? a.shape.slice(0, -2) : b.shape.slice(0, -2);
  const outShape = outShapeOuterDims.concat([outerShapeA, outerShapeB]);

  util.assert(
      innerShapeA === innerShapeB,
      () => `Error in matMul: inner shapes (${innerShapeA}) and (` +
          `${innerShapeB}) of Tensors with shapes ${a.shape} and ` +
          `${b.shape} and transposeA=${transposeA}` +
          ` and transposeB=${transposeB} must match.`);

  const a3dShape = transposeA ? [batchDimA, innerShapeA, outerShapeA] :
                                [batchDimA, outerShapeA, innerShapeA];
  const b3dShape = transposeB ? [batchDimB, outerShapeB, innerShapeB] :
                                [batchDimB, innerShapeB, outerShapeB];

  // The rest of the implementation is designed to operate on rank-3 tensors
  const a3d = reshape({inputs: {x: a}, backend, attrs: {shape: a3dShape}});
  const b3d = reshape({inputs: {x: b}, backend, attrs: {shape: b3dShape}});

  const a3dId = backend.dataIdMap.get(a3d.dataId).id;
  const b3dId = backend.dataIdMap.get(b3d.dataId).id;

  const leftDim = transposeA ? a3d.shape[2] : a3d.shape[1];
  const rightDim = transposeB ? b3d.shape[1] : b3d.shape[2];
  const batchDim = Math.max(batchDimA, batchDimB);

  const out = backend.makeOutput([batchDim, leftDim, rightDim], a3d.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;

  const aShapeBytes = new Uint8Array(new Int32Array(a3d.shape).buffer);
  const bShapeBytes = new Uint8Array(new Int32Array(b3d.shape).buffer);

  wasmBatchMatMul(
      a3dId, aShapeBytes, a3d.shape.length, b3dId, bShapeBytes,
      b3d.shape.length, transposeA, transposeB, outId);

  backend.disposeData(a3d.dataId);
  backend.disposeData(b3d.dataId);

  out.shape = outShape;
  return out;
}

export const batchMatMulConfig: KernelConfig = {
  kernelName: BatchMatMul,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: batchMatMul as {} as KernelFunc
};
