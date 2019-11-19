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

let wasmFunc: (
    boxesId: number, scoresId: number, maxOutputSize: number,
    iouThreshold: number, scoreThreshold: number) => Uint8Array;

function setup(backend: BackendWasm): void {
  wasmFunc = backend.wasm.cwrap(
      'NonMaxSuppressionV3',
      'array',  // memoryOffset
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

  const resultBytes =
      wasmFunc(boxesId, scoresId, maxOutputSize, iouThreshold, scoreThreshold);
  const result = new Int32Array(resultBytes.buffer);
  const memOffset = result[0];
  const size = result[1];
  const outShape = [size];
  return backend.makeOutput(outShape, 'int32', memOffset);
}

registerKernel({
  kernelName: 'NonMaxSuppressionV3',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc,
});
