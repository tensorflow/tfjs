/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {KernelConfig, KernelFunc, NonMaxSuppressionV4, NonMaxSuppressionV4Attrs, NonMaxSuppressionV4Inputs, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {parseResultStruct} from './NonMaxSuppression_util';

let wasmFunc: (
    boxesId: number, scoresId: number, maxOutputSize: number,
    iouThreshold: number, scoreThreshold: number,
    padToMaxOutputSize: boolean) => number;

function setup(backend: BackendWasm): void {
  wasmFunc = backend.wasm.cwrap(
      NonMaxSuppressionV4,
      'number',  // Result*
      [
        'number',  // boxesId
        'number',  // scoresId
        'number',  // maxOutputSize
        'number',  // iouThreshold
        'number',  // scoreThreshold
        'bool',    // padToMaxOutputSize
      ]);
}

function nonMaxSuppressionV4(args: {
  backend: BackendWasm,
  inputs: NonMaxSuppressionV4Inputs,
  attrs: NonMaxSuppressionV4Attrs
}): TensorInfo[] {
  const {backend, inputs, attrs} = args;
  const {iouThreshold, maxOutputSize, scoreThreshold, padToMaxOutputSize} =
      attrs;
  const {boxes, scores} = inputs;

  const boxesId = backend.dataIdMap.get(boxes.dataId).id;
  const scoresId = backend.dataIdMap.get(scores.dataId).id;

  const resOffset = wasmFunc(
      boxesId, scoresId, maxOutputSize, iouThreshold, scoreThreshold,
      padToMaxOutputSize);

  const {pSelectedIndices, selectedSize, pSelectedScores, pValidOutputs} =
      parseResultStruct(backend, resOffset);

  // Since we are not using scores for V4, we have to delete it from the heap.
  backend.wasm._free(pSelectedScores);

  const selectedIndicesTensor =
      backend.makeOutput([selectedSize], 'int32', pSelectedIndices);

  const validOutputsTensor = backend.makeOutput([], 'int32', pValidOutputs);

  return [selectedIndicesTensor, validOutputsTensor];
}

export const nonMaxSuppressionV4Config: KernelConfig = {
  kernelName: NonMaxSuppressionV4,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: nonMaxSuppressionV4 as {} as KernelFunc,
};
