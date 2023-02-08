/**
 * @license
 * Copyright 2023 Google LLC.
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

import {KernelConfig, KernelFunc, SearchSorted, SearchSortedAttrs, SearchSortedInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';
import {CppDType} from './types';

let wasmSearchSorted: (
    sortedSequenceId: number, valuesId: number, batchSize: number,
    sequenceSize: number, valuesSize: number, dtype: number,
    isSideLeft: boolean, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmSearchSorted = backend.wasm.cwrap(SearchSorted, null /* void */, [
    'number',  // sortedSequenceId
    'number',  // valuesId
    'number',  // batchSize
    'number',  // sequenceSize
    'number',  // valuesSize
    'number',  // dtype
    'bool',    // isSideLeft
    'number',  // outId
  ]);
}

function searchSorted(args: {
  inputs: SearchSortedInputs,
  backend: BackendWasm,
  attrs: SearchSortedAttrs,
}) {
  const {inputs, backend, attrs} = args;
  const {sortedSequence, values} = inputs;
  const {side} = attrs;

  if (sortedSequence.dtype !== values.dtype) {
    throw new Error(
        `SearchSorted error: sorted_sequence must have the same dtype as values. Got ${
            sortedSequence.dtype} and ${values.dtype}`);
  }

  const out = backend.makeOutput(values.shape, 'int32');

  function tensorId(x: TensorInfo) {
    return backend.dataIdMap.get(x.dataId).id!;
  }
  wasmSearchSorted(
      tensorId(sortedSequence),
      tensorId(values),
      /*batchSize=*/sortedSequence.shape[0],
      /*sequenceSize=*/sortedSequence.shape[1],
      /*valuesSize=*/values.shape[1],
      /*dtype=*/CppDType[sortedSequence.dtype],
      /*isSideLeft=*/side === 'left',
      tensorId(out),
  );

  return out;
}

export const searchSortedConfig: KernelConfig = {
  kernelName: SearchSorted,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: searchSorted as unknown as KernelFunc
};
