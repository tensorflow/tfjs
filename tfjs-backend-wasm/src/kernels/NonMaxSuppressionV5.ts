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
  softNmsSigma: number;
}

// Analogous to `struct Result` in `NonMaxSuppressionV5.cc`.
interface Result {
  pSelectedIndices: number;
  selectedSize: number;
  pSelectedScores: number;
}

/**
 * Parse the result of the c++ method, which is a data structure with four ints
 * for selected_indices pointer, selected_indices size, selected_scores pointer,
 * selected_scores size.
 */
function parseResultStruct(backend: BackendWasm, resOffset: number): Result {
  const result = new Int32Array(backend.wasm.HEAPU8.buffer, resOffset, 3);
  const pSelectedIndices = result[0];
  const selectedSize = result[1];
  const pSelectedScores = result[2];

  // Since the result was allocated on the heap, we have to delete it.
  backend.wasm._free(resOffset);

  return {pSelectedIndices, selectedSize, pSelectedScores};
}

let wasmFunc:
    (boxesId: number, scoresId: number, maxOutputSize: number,
     iouThreshold: number, scoreThreshold: number, softNmsSigma: number) =>
        number;

function setup(backend: BackendWasm): void {
  wasmFunc = backend.wasm.cwrap(
      'NonMaxSuppressionV5',
      'number',  // Result*
      [
        'number',  // boxesId
        'number',  // scoresId
        'number',  // maxOutputSize
        'number',  // iouThreshold
        'number',  // scoreThreshold
        'number',  // softNmsSigma
      ]);
}

function kernelFunc(args: {
  backend: BackendWasm,
  inputs: NonMaxSuppressionInputs,
  attrs: NonMaxSuppressionAttrs
}): TensorInfo[] {
  const {backend, inputs, attrs} = args;
  const {iouThreshold, maxOutputSize, scoreThreshold, softNmsSigma} = attrs;
  const {boxes, scores} = inputs;

  const boxesId = backend.dataIdMap.get(boxes.dataId).id;
  const scoresId = backend.dataIdMap.get(scores.dataId).id;

  const resOffset = wasmFunc(
      boxesId, scoresId, maxOutputSize, iouThreshold, scoreThreshold,
      softNmsSigma);

  const {
    pSelectedIndices,
    selectedSize,
    pSelectedScores,
  } = parseResultStruct(backend, resOffset);

  const selectedIndices =
      backend.makeOutput([selectedSize], 'int32', pSelectedIndices);
  const selectedScores =
      backend.makeOutput([selectedSize], 'float32', pSelectedScores);

  return [selectedIndices, selectedScores];
}

registerKernel({
  kernelName: 'NonMaxSuppressionV5',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc,
});
