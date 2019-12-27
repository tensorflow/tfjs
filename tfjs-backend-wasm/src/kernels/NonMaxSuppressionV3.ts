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

import {NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface NonMaxSuppressionInputs extends NamedTensorInfoMap {
  boxes: TensorInfo;
  scores: TensorInfo;
}

interface NonMaxSuppressionAttrs extends NamedAttrMap {
  maxOutputSize: number;
  iouThreshold: number;
  scoreThreshold: number;
}

// Analogous to `struct Result` in `NonMaxSuppressionV3.cc`.
interface Result {
  memOffset: number;
  size: number;
}

/**
 * Parse the result of the c++ method, which is a data structure with two ints
 * (memOffset and size).
 */
function parseResultStruct(backend: BackendWasm, resOffset: number): Result {
  // The result of c++ method is a data structure with two ints (memOffset, and
  // size).
  const result = new Int32Array(backend.wasm.HEAPU8.buffer, resOffset, 2);
  const memOffset = result[0];
  const size = result[1];
  // Since the result was allocated on the heap, we have to delete it.
  backend.wasm._free(resOffset);
  return {memOffset, size};
}

let wasmFunc: (
    boxesId: number, scoresId: number, maxOutputSize: number,
    iouThreshold: number, scoreThreshold: number) => number;

function setup(backend: BackendWasm): void {
  wasmFunc = backend.wasm.cwrap(
      'NonMaxSuppressionV3',
      'number',  // Result*
      [
        'number',  // boxesId
        'number',  // scoresId
        'number',  // maxOutputSize
        'number',  // iouThreshold
        'number',  // scoreThreshold
      ]);
}

function kernelFunc(args: {
  backend: BackendWasm,
  inputs: NonMaxSuppressionInputs,
  attrs: NonMaxSuppressionAttrs
}): TensorInfo {
  const {backend, inputs, attrs} = args;
  const {iouThreshold, maxOutputSize, scoreThreshold} = attrs;
  const {boxes, scores} = inputs;

  const boxesId = backend.dataIdMap.get(boxes.dataId).id;
  const scoresId = backend.dataIdMap.get(scores.dataId).id;

  const resOffset =
      wasmFunc(boxesId, scoresId, maxOutputSize, iouThreshold, scoreThreshold);

  const {memOffset, size} = parseResultStruct(backend, resOffset);

  const outShape = [size];
  return backend.makeOutput(outShape, 'int32', memOffset);
}

registerKernel({
  kernelName: 'NonMaxSuppressionV3',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc,
});
