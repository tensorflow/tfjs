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

import {KernelFunc, registerKernel, Rotate, RotateAttrs, RotateInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmRotate: (
    xId: number, batch: number, imageHeight: number, imageWidth: number,
    numChannels: number, radians: number, centerX: number, centerY: number,
    fillBytes: Uint8Array, fillLength: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmRotate = backend.wasm.cwrap(Rotate, null /* void */, [
    'number',  // xId
    'number',  // batch
    'number',  // imageHeight
    'number',  // imageWidth
    'number',  // numChannels
    'number',  // radians
    'number',  // centerX
    'number',  // centerY
    'array',   // fillBytes
    'number',  // fillLength
    'number',  // outId
  ]);
}

export function rotate(
    args: {inputs: RotateInputs, backend: BackendWasm, attrs: RotateAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {image} = inputs;
  const {radians, fillValue, center} = attrs;

  const out = backend.makeOutput(image.shape, image.dtype);
  const imageId = backend.dataIdMap.get(image.dataId).id;
  const outId = backend.dataIdMap.get(out.dataId).id;

  const [batch, imageHeight, imageWidth, numChannels] = image.shape;

  const centerX =
      imageWidth * (typeof center === 'number' ? center : center[0]);
  const centerY =
      imageHeight * (typeof center === 'number' ? center : center[1]);

  const fillValues = typeof fillValue === 'number' ?
      [fillValue, fillValue, fillValue] :
      fillValue;
  const fillBytes = new Uint8Array(new Int32Array(fillValues).buffer);

  wasmRotate(
      imageId, batch, imageHeight, imageWidth, numChannels, radians, centerX,
      centerY, fillBytes, fillValues.length, outId);
  return out;
}

registerKernel({
  kernelName: Rotate,
  backendName: 'wasm',
  kernelFunc: rotate as KernelFunc,
  setupFunc: setup,
});
