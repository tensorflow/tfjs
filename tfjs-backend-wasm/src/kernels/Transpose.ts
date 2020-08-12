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

import {KernelConfig, KernelFunc, TensorInfo, Transpose, TransposeAttrs, TransposeInputs} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {identity} from './Identity';
import {CppDType} from './types';

let wasmTranspose: (
    xId: number, xShape: Uint8Array, xShapeLength: number, dtype: CppDType,
    outId: number, perm: Uint8Array, permLength: number) => void;

function setup(backend: BackendWasm) {
  wasmTranspose = backend.wasm.cwrap(Transpose, null /* void */, [
    'number',  // xId
    'array',   // x.shape
    'number',  // x.shape.length
    'number',  // dtype
    'number',  // outId
    'array',   // perm
    'number',  // perm.length
  ]);
}

export function transpose(
    args:
        {inputs: TransposeInputs, backend: BackendWasm, attrs: TransposeAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  // Reduce any dimensions with size one. Lower-rank transpose kernel performs
  // better due to simpler memory access pattern.
  const [reducedShape, perm] = removeOneSizeDims(inputs.x.shape, attrs.perm);

  let permIsNoOp = true;
  for (let i = 0; i < perm.length; i++) {
    if (perm[i] !== i) {
      permIsNoOp = false;
    }
  }
  const outShape = computeOutShape(inputs.x.shape, attrs.perm);
  const x = {
    dataId: inputs.x.dataId,
    shape: reducedShape,
    dtype: inputs.x.dtype
  };

  if (permIsNoOp) {
    const cloned = identity({inputs, backend});
    cloned.shape = outShape;
    return cloned;
  }

  const out = backend.makeOutput(outShape, x.dtype);
  const xId = backend.dataIdMap.get(x.dataId).id;
  const outId = backend.dataIdMap.get(out.dataId).id;
  const permBytes = new Uint8Array(new Int32Array(perm).buffer);
  const xShapeBytes = new Uint8Array(new Int32Array(x.shape).buffer);

  wasmTranspose(
      xId, xShapeBytes, x.shape.length, CppDType[x.dtype], outId, permBytes,
      perm.length);
  return out;
}

function computeOutShape(inShape: number[], perm: number[]): number[] {
  const outShape = new Array(inShape.length);
  for (let i = 0; i < outShape.length; i++) {
    outShape[i] = inShape[perm[i]];
  }
  return outShape;
}

function removeOneSizeDims(
    shape: number[], perm: number[]): [number[], number[]] {
  const newShape: number[] = [];
  const newPerm: number[] = [];
  for (let i = 0; i < shape.length; ++i) {
    if (shape[i] !== 1) {
      newShape.push(shape[i]);
    }
    if (shape[perm[i]] !== 1) {
      newPerm.push(perm[i]);
    }
  }
  for (let i = 0; i < newPerm.length; ++i) {
    let minValIdx = -1;
    for (let j = 0; j < newPerm.length; ++j) {
      if (newPerm[j] >= i &&
          (minValIdx === -1 || newPerm[minValIdx] > newPerm[j])) {
        minValIdx = j;
      }
    }
    newPerm[minValIdx] = i;
  }
  return [newShape, newPerm];
}

export const transposeConfig: KernelConfig = {
  kernelName: Transpose,
  backendName: 'wasm',
  kernelFunc: transpose as {} as KernelFunc,
  setupFunc: setup,
};
