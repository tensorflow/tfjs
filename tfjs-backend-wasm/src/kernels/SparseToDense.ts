/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {backend_util, KernelConfig, KernelFunc, SparseToDense, SparseToDenseAttrs, SparseToDenseInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {CppDType} from './types';

let wasmSparseToDense: (
    sparseIndicesId: number, sparseValuesId: number, sparseValuesRank: number,
    defaultValueId: number, dtype: CppDType, sliceRank: number,
    numUpdates: number, sliceSize: number, strides: Uint8Array,
    outputSize: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmSparseToDense = backend.wasm.cwrap(SparseToDense, null /*void*/, [
    'number',  // sparseIndicesId
    'number',  // sparseValuesId
    'number',  // sparseValuesRank
    'number',  // defaultValueId
    'number',  // dtype
    'number',  // sliceRank
    'number',  // numUpdates
    'number',  // sliceSize
    'array',   // strides
    'number',  // outputSize
    'number',  // outId
  ]);
}

function sparseToDense(args: {
  backend: BackendWasm,
  inputs: SparseToDenseInputs,
  attrs: SparseToDenseAttrs
}): TensorInfo {
  const {backend, inputs, attrs} = args;
  const {sparseIndices, sparseValues, defaultValue} = inputs;
  const {outputShape} = attrs;

  const out = backend.makeOutput(outputShape, defaultValue.dtype);
  if (util.sizeFromShape(outputShape) === 0) {
    return out;
  }

  const {sliceRank, numUpdates, sliceSize, strides, outputSize} =
      backend_util.calculateShapes(sparseValues, sparseIndices, outputShape);

  const sparseIndicesId = backend.dataIdMap.get(sparseIndices.dataId).id;
  const sparseValuesId = backend.dataIdMap.get(sparseValues.dataId).id;
  const defaultValueId = backend.dataIdMap.get(defaultValue.dataId).id;

  const stridesBytes = new Uint8Array(new Int32Array(strides).buffer);

  const outId = backend.dataIdMap.get(out.dataId).id;

  wasmSparseToDense(
      sparseIndicesId, sparseValuesId, sparseValues.shape.length,
      defaultValueId, CppDType[defaultValue.dtype], sliceRank, numUpdates,
      sliceSize, stridesBytes, outputSize, outId);

  return out;
}

export const sparseToDenseConfig: KernelConfig = {
  kernelName: SparseToDense,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: sparseToDense as unknown as KernelFunc
};
