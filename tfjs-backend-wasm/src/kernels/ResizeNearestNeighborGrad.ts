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

import {KernelConfig, KernelFunc, ResizeNearestNeighborGrad, ResizeNearestNeighborGradAttrs, ResizeNearestNeighborGradInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {cast} from './Cast';

let wasmResizeNearestNeighborGrad: (
    imagesId: number, dyId: number, dxId: number, imagesShape: Uint8Array,
    dyShape: Uint8Array, alignCorners: boolean) => void;

function setup(backend: BackendWasm): void {
  wasmResizeNearestNeighborGrad = backend.wasm.cwrap(
      ResizeNearestNeighborGrad, null /*void*/,
      [
        'number',   // imagesId
        'number',   // dyId
        'number',   // dxId
        'array',    // imagesShape
        'array',    // dyShape
        'boolean',  // alignCorners
      ]);
}

function resizeNearestNeighborGrad(args: {
  backend: BackendWasm; inputs: ResizeNearestNeighborGradInputs;
  attrs: ResizeNearestNeighborGradAttrs;
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {images, dy} = inputs;
  const {alignCorners} = attrs;

  const dx = backend.makeOutput(images.shape, 'float32');

  let xData = backend.dataIdMap.get(images.dataId);
  let castedData;
  if (xData.dtype !== 'float32') {
    castedData = cast({
      backend,
      inputs: {x: images},
      attrs: {dtype: 'float32'},
    });
    xData = backend.dataIdMap.get(castedData.dataId);
  }

  wasmResizeNearestNeighborGrad(
      backend.dataIdMap.get(images.dataId).id,
      backend.dataIdMap.get(dy.dataId).id,
      backend.dataIdMap.get(dx.dataId).id,
      new Uint8Array(new Int32Array(images.shape).buffer),
      new Uint8Array(new Int32Array(dy.shape).buffer),
      alignCorners,
  );

  if (castedData != null) {
    backend.disposeData(castedData.dataId);
  }

  return dx;
}

export const resizeNearestNeighborGradConfig: KernelConfig = {
  kernelName: ResizeNearestNeighborGrad,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: resizeNearestNeighborGrad as unknown as KernelFunc,
};
