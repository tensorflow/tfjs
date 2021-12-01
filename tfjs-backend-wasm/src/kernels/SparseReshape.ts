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

import {backend_util, KernelConfig, KernelFunc, SparseReshape, SparseReshapeInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmSparseReshape: (
    inputIndicesId: number, inputShapeId: number, newShapeId: number,
    nnz: number, newIndicesId: number, outputShapeId: number,
    exceptionValuesId: number) => void;

function setup(backend: BackendWasm): void {
  wasmSparseReshape = backend.wasm.cwrap(SparseReshape, null /*void*/, [
    'number',  // inputIndicesId
    'number',  // inputShapeId
    'number',  // newShapeId
    'number',  // nnz
    'number',  // newIndicesId
    'number',  // outputShapeId
    'number',  // exceptionValuesId
  ]);
}

function sparseReshape(args: {
  backend: BackendWasm,
  inputs: SparseReshapeInputs,
}): [TensorInfo, TensorInfo] {
  const {backend, inputs} = args;
  const {inputIndices, inputShape, newShape} = inputs;

  if (inputIndices.shape.length !== 2) {
    throw new Error(`Input indices should be a matrix but received shape
        ${inputIndices.shape}`);
  }
  if (inputShape.shape.length !== 1) {
    throw new Error(`Input shape should be a vector but received shape
        ${inputShape.shape}`);
  }
  if (newShape.shape.length !== 1) {
    throw new Error(
        `Target shape should be a vector but received shape ${newShape.shape}`);
  }

  const inputIndicesId = backend.dataIdMap.get(inputIndices.dataId).id;
  const inputShapeId = backend.dataIdMap.get(inputShape.dataId).id;
  const newShapeId = backend.dataIdMap.get(newShape.dataId).id;

  const nnz = inputIndices.shape[0];
  const outputRank = util.sizeFromShape(newShape.shape);

  const newIndices = backend.makeOutput([nnz, outputRank], inputIndices.dtype);
  const newIndicesId = backend.dataIdMap.get(newIndices.dataId).id;

  const outputShape = backend.makeOutput([outputRank], newShape.dtype);
  const outputShapeId = backend.dataIdMap.get(outputShape.dataId).id;

  const exceptionValues = backend.makeOutput([3], 'int32');
  const exceptionValuesId = backend.dataIdMap.get(exceptionValues.dataId).id;

  wasmSparseReshape(
      inputIndicesId, inputShapeId, newShapeId, nnz, newIndicesId,
      outputShapeId, exceptionValuesId);

  const exceptionValuesArray =
      backend.readSync(exceptionValues.dataId) as Int32Array;

  let exceptionMessage: string;
  switch (exceptionValuesArray[0]) {
    case 0: {
      exceptionMessage =
          backend_util.getSparseReshapeMultipleNegativeOneOutputDimErrorMessage(
              exceptionValuesArray[1], exceptionValuesArray[2]);
      break;
    }
    case 1: {
      exceptionMessage =
          backend_util.getSparseReshapeNegativeOutputDimErrorMessage(
              exceptionValuesArray[1], exceptionValuesArray[2]);
      break;
    }
    case 2:
      exceptionMessage =
          backend_util.getSparseReshapeEmptyTensorZeroOutputDimErrorMessage();
      break;
    case 3: {
      const inputShapeValues =
          Array.from(backend.readSync(inputShape.dataId) as Int32Array),
            outputShapeValues =
                Array.from(backend.readSync(outputShape.dataId) as Int32Array);
      exceptionMessage =
          backend_util.getSparseReshapeInputOutputMultipleErrorMessage(
              inputShapeValues, outputShapeValues);
      break;
    }
    case 4: {
      const inputShapeValues =
          Array.from(backend.readSync(inputShape.dataId) as Int32Array),
            outputShapeValues =
                Array.from(backend.readSync(outputShape.dataId) as Int32Array);
      exceptionMessage =
          backend_util.getSparseReshapeInputOutputMismatchErrorMessage(
              inputShapeValues, outputShapeValues);
      break;
    }
    default:
      exceptionMessage = '';
  }

  backend.disposeData(exceptionValues.dataId);
  if (exceptionMessage) {
    backend.disposeData(newIndices.dataId);
    backend.disposeData(outputShape.dataId);
    throw new Error(exceptionMessage);
  }

  return [newIndices, outputShape];
}

export const sparseReshapeConfig: KernelConfig = {
  kernelName: SparseReshape,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: sparseReshape as {} as KernelFunc
};
