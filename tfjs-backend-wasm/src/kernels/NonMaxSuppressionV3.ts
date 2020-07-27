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

import {KernelConfig, KernelFunc, NonMaxSuppressionV3, NonMaxSuppressionV3Attrs, NonMaxSuppressionV3Inputs, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {parseResultStruct} from './NonMaxSuppression_util';

let wasmFunc: (
    boxesId: number, scoresId: number, maxOutputSize: number,
    iouThreshold: number, scoreThreshold: number) => number;

function setup(backend: BackendWasm): void {
  wasmFunc = backend.wasm.cwrap(
      NonMaxSuppressionV3,
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
  inputs: NonMaxSuppressionV3Inputs,
  attrs: NonMaxSuppressionV3Attrs
}): TensorInfo {
  const {backend, inputs, attrs} = args;
  const {iouThreshold, maxOutputSize, scoreThreshold} = attrs;
  const {boxes, scores} = inputs;

  const boxesId = backend.dataIdMap.get(boxes.dataId).id;
  const scoresId = backend.dataIdMap.get(scores.dataId).id;

  const resOffset =
      wasmFunc(boxesId, scoresId, maxOutputSize, iouThreshold, scoreThreshold);

  const {pSelectedIndices, selectedSize, pSelectedScores, pValidOutputs} =
      parseResultStruct(backend, resOffset);

  // Since we are not using scores for V3, we have to delete it from the heap.
  backend.wasm._free(pSelectedScores);
  backend.wasm._free(pValidOutputs);

  const selectedIndicesTensor =
      backend.makeOutput([selectedSize], 'int32', pSelectedIndices);

  return selectedIndicesTensor;
}

export const nonMaxSuppressionV3Config: KernelConfig = {
  kernelName: NonMaxSuppressionV3,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: kernelFunc as {} as KernelFunc,
};
