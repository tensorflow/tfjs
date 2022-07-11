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

import {backend_util, KernelConfig, KernelFunc, SparseFillEmptyRows, SparseFillEmptyRowsInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';
import {slice} from './Slice';

import {CppDType} from './types';

let wasmSparseFillEmptyRows: (
    indicesId: number, valuesId: number, valuesDType: number,
    indicesCount: number, denseRows: number, rank: number,
    defaultValueId: number, outputIndicesId: number, outputValuesId: number,
    emptyRowIndicatorId: number, reverseIndexMapId: number,
    exceptionValuesId: number) => number;

export function setup(backend: BackendWasm): void {
  wasmSparseFillEmptyRows =
      backend.wasm.cwrap('SparseFillEmptyRows', 'number', [
        'number',  // indicesId
        'number',  // valuesId
        'number',  // valuesDType
        'number',  // indicesCount
        'number',  // denseRows
        'number',  // rank
        'number',  // defaultValueId
        'number',  // outputIndicesId
        'number',  // outputValuesId
        'number',  // emptyRowIndicatorId
        'number',  // reverseIndexMapId
        'number',  // exceptionValuesId
      ]);
}

export function sparseFillEmptyRows(args: {
  backend: BackendWasm,
  inputs: SparseFillEmptyRowsInputs,
}): [TensorInfo, TensorInfo, TensorInfo, TensorInfo] {
  const {backend, inputs} = args;
  const {indices, values, denseShape, defaultValue} = inputs;

  const indicesCount = indices.shape[0];
  const rank = indices.shape[1];
  const denseRows = backend.readSync(denseShape.dataId)[0] as number;

  // Set output size to maximum possible and resize later (actual result
  // might be smaller).
  const maxOutputIndicesShape = [indicesCount + denseRows, rank];

  const indicesId = backend.dataIdMap.get(indices.dataId).id;
  const valuesId = backend.dataIdMap.get(values.dataId).id;
  const defaultValueId = backend.dataIdMap.get(defaultValue.dataId).id;

  const outputIndices =
      backend.makeOutput(maxOutputIndicesShape, indices.dtype);
  const outputIndicesId = backend.dataIdMap.get(outputIndices.dataId).id;

  const outputValues =
      backend.makeOutput(maxOutputIndicesShape.slice(0, 1), values.dtype);
  const outputValuesId = backend.dataIdMap.get(outputValues.dataId).id;

  const emptyRowIndicator = backend.makeOutput([denseRows], 'bool');
  const emptyRowIndicatorId =
      backend.dataIdMap.get(emptyRowIndicator.dataId).id;

  const reverseIndexMap = backend.makeOutput([indicesCount], indices.dtype);
  const reverseIndexMapId = backend.dataIdMap.get(reverseIndexMap.dataId).id;

  const exceptionValues = backend.makeOutput([4], 'int32');
  const exceptionValuesId = backend.dataIdMap.get(exceptionValues.dataId).id;

  const outputRows = wasmSparseFillEmptyRows(
      indicesId, valuesId, CppDType[values.dtype], indicesCount, denseRows,
      rank, defaultValueId, outputIndicesId, outputValuesId,
      emptyRowIndicatorId, reverseIndexMapId, exceptionValuesId);

  const exceptionValuesArray =
      backend.readSync(exceptionValues.dataId) as Int32Array;

  let exceptionMessage: string;
  switch (exceptionValuesArray[0]) {
    case 1: {
      exceptionMessage =
          backend_util.getSparseFillEmptyRowsIndicesDenseShapeMismatch(
              exceptionValuesArray[1]);
      break;
    }
    case 2: {
      exceptionMessage =
          backend_util.getSparseFillEmptyRowsNegativeIndexErrorMessage(
              exceptionValuesArray[1], exceptionValuesArray[2]);
      break;
    }
    case 3:
      exceptionMessage =
          backend_util.getSparseFillEmptyRowsOutOfRangeIndexErrorMessage(
              exceptionValuesArray[1], exceptionValuesArray[2],
              exceptionValuesArray[3]);
      break;
    default:
      exceptionMessage = '';
  }

  backend.disposeData(exceptionValues.dataId);
  if (exceptionMessage) {
    backend.disposeData(outputIndices.dataId);
    backend.disposeData(outputValues.dataId);
    backend.disposeData(emptyRowIndicator.dataId);
    backend.disposeData(reverseIndexMap.dataId);
    throw new Error(exceptionMessage);
  }

  let resizedIndices = outputIndices;
  let resizedValues = outputValues;
  // Overestimated output size.
  if (outputRows !== maxOutputIndicesShape[0]) {
    resizedIndices = slice({
      inputs: {x: outputIndices},
      attrs: {begin: 0, size: [outputRows, rank]},
      backend
    });
    resizedValues = slice({
      inputs: {x: outputValues},
      attrs: {begin: 0, size: outputRows},
      backend
    });
    backend.disposeData(outputIndices.dataId);
    backend.disposeData(outputValues.dataId);
  }

  return [resizedIndices, resizedValues, emptyRowIndicator, reverseIndexMap];
}

export const sparseFillEmptyRowsConfig: KernelConfig = {
  kernelName: SparseFillEmptyRows,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: sparseFillEmptyRows as {} as KernelFunc
};
