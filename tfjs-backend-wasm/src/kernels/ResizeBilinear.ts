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

import {KernelConfig, KernelFunc, ResizeBilinear, ResizeBilinearAttrs, ResizeBilinearInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {cast} from './Cast';

let wasmResizeBilinear: (
    xId: number, batch: number, oldHeight: number, oldWidth: number,
    numChannels: number, newHeight: number, newWidth: number,
    alignCorners: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmResizeBilinear = backend.wasm.cwrap(ResizeBilinear, null /*void*/, [
    'number',  // xId
    'number',  // batch
    'number',  // oldHeight
    'number',  // oldWidth
    'number',  // numChannels
    'number',  // newHeight
    'number',  // newWidth
    'number',  // alignCorners
    'number'   // outId
  ]);
}

function resizeBilinear(args: {
  backend: BackendWasm,
  inputs: ResizeBilinearInputs,
  attrs: ResizeBilinearAttrs
}): TensorInfo {
  const {backend, inputs, attrs} = args;

  const {images} = inputs;
  const {alignCorners, size} = attrs;
  const [newHeight, newWidth] = size;

  const [batch, oldHeight, oldWidth, numChannels] = images.shape;
  const outShape = [batch, newHeight, newWidth, numChannels];

  let xData = backend.dataIdMap.get(images.dataId);
  let castedData;
  if (xData.dtype !== 'float32') {
    castedData =
        cast({backend, inputs: {x: images}, attrs: {dtype: 'float32'}});
    xData = backend.dataIdMap.get(castedData.dataId);
  }
  const xId = xData.id;

  const out = backend.makeOutput(outShape, 'float32');
  if (util.sizeFromShape(images.shape) === 0) {
    return out;
  }
  const outId = backend.dataIdMap.get(out.dataId).id;

  wasmResizeBilinear(
      xId, batch, oldHeight, oldWidth, numChannels, newHeight, newWidth,
      alignCorners ? 1 : 0, outId);

  if (castedData != null) {
    backend.disposeData(castedData.dataId);
  }

  return out;
}

export const resizeBilinearConfig: KernelConfig = {
  kernelName: ResizeBilinear,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: resizeBilinear as {} as KernelFunc
};
