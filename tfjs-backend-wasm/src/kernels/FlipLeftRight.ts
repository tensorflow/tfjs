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

import {FlipLeftRight, FlipLeftRightInputs, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmFlipLeftRight: (
    xId: number, batch: number, imageHeight: number, imageWidth: number,
    numChannels: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmFlipLeftRight = backend.wasm.cwrap(FlipLeftRight, null /* void */, [
    'number',  // xId
    'number',  // batch
    'number',  // imageHeight
    'number',  // imageWidth
    'number',  // numChannels
    'number',  // outId
  ]);
}

export function flipLeftRight(
    args: {inputs: FlipLeftRightInputs, backend: BackendWasm}): TensorInfo {
  const {inputs, backend} = args;
  const {image} = inputs;

  const out = backend.makeOutput(image.shape, image.dtype);
  const imageId = backend.dataIdMap.get(image.dataId).id;
  const outId = backend.dataIdMap.get(out.dataId).id;

  const [batch, imageHeight, imageWidth, numChannels] = image.shape;

  wasmFlipLeftRight(
      imageId, batch, imageHeight, imageWidth, numChannels, outId);
  return out;
}

export const flipLeftRightConfig: KernelConfig = {
  kernelName: FlipLeftRight,
  backendName: 'wasm',
  kernelFunc: flipLeftRight as {} as KernelFunc,
  setupFunc: setup
};
