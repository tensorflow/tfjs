/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {
  KernelConfig,
  KernelFunc,
  ResizeNearestNeighbor,
  ResizeNearestNeighborAttrs,
  ResizeNearestNeighborInputs,
  TensorInfo,
  util,
} from '@tensorflow/tfjs-core';

import { BackendWasm } from '../backend_wasm';

import { cast } from './Cast';

let wasmResizeNearestNeighbor: (
  xId: number,
  batch: number,
  oldHeight: number,
  oldWidth: number,
  numChannels: number,
  newHeight: number,
  newWidth: number,
  alignCorners: number,
  halfPixelCenters: number,
  outId: number
) => void;

function setup(backend: BackendWasm): void {
  wasmResizeNearestNeighbor = backend.wasm.cwrap(
    ResizeNearestNeighbor,
    null /*void*/,
    [
      'number', // xId
      'number', // batch
      'number', // oldHeight
      'number', // oldWidth
      'number', // numChannels
      'number', // newHeight
      'number', // newWidth
      'number', // alignCorners
      'number', // halfPixelCenters
      'number', // outId
    ]
  );
}

function resizeNearestNeighbor(args: {
  backend: BackendWasm;
  inputs: ResizeNearestNeighborInputs;
  attrs: ResizeNearestNeighborAttrs;
}): TensorInfo {
  const { backend, inputs, attrs } = args;
  const { images } = inputs;
  const { alignCorners, halfPixelCenters, size } = attrs;

  const [newHeight, newWidth] = size;

  const [batch, oldHeight, oldWidth, numChannels] = images.shape;
  const outShape = [batch, newHeight, newWidth, numChannels];

  const out = backend.makeOutput(outShape, 'float32');
  if (util.sizeFromShape(images.shape) === 0) {
    return out;
  }

  let xData = backend.dataIdMap.get(images.dataId);
  let castedData;
  if (xData.dtype !== 'float32') {
    castedData = cast({
      backend,
      inputs: { x: images },
      attrs: { dtype: 'float32' },
    });
    xData = backend.dataIdMap.get(castedData.dataId);
  }

  const xId = xData.id;
  const outId = backend.dataIdMap.get(out.dataId).id;

  wasmResizeNearestNeighbor(
    xId,
    batch,
    oldHeight,
    oldWidth,
    numChannels,
    newHeight,
    newWidth,
    alignCorners ? 1 : 0,
    halfPixelCenters ? 1 : 0,
    outId
  );

  if (castedData != null) {
    backend.disposeData(castedData.dataId);
  }

  return out;
}

export const resizeNearestNeighborConfig: KernelConfig = {
  kernelName: ResizeNearestNeighbor,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: resizeNearestNeighbor as unknown as KernelFunc,
};
