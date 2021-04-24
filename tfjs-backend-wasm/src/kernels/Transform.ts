/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {KernelConfig, KernelFunc, TensorInfo, Transform, TransformAttrs, TransformInputs, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmTransform: (
    imageId: number, transformsId: number, isBatchTransform: boolean,
    batch: number, outHeight: number, outWidth: number, numChannels: number,
    imageWidth: number, imageHeight: number, strides: Uint8Array,
    stridesLength: number, interpolationModeId: number, fillModeId: number,
    fillValue: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmTransform = backend.wasm.cwrap(Transform, null /*void*/, [
    'number',  // imageId
    'number',  // transformsId
    'bool',    // isBatchTransform
    'number',  // batch
    'number',  // outHeight
    'number',  // outWidth
    'number',  // numChannels
    'number',  // imageWidth
    'number',  // imageHeight
    'array',   // strides
    'number',  // stridesLength
    'number',  // interpolationModeId
    'number',  // fillModeId
    'number',  // fillValue
    'number'   // outId
  ]);
}

function transform(
    args:
        {backend: BackendWasm, inputs: TransformInputs, attrs: TransformAttrs}):
    TensorInfo {
  const {backend, inputs, attrs} = args;
  const {image, transforms} = inputs;
  const {interpolation, fillMode, fillValue, outputShape} = attrs;

  const [batch, imageHeight, imageWidth, numChannels] = image.shape;
  const [outHeight, outWidth] =
      outputShape != null ? outputShape : [imageHeight, imageWidth];
  const outShape =
      [batch, outHeight, outWidth,
       numChannels] as [number, number, number, number];
  const strides =
      new Uint8Array(new Int32Array(util.computeStrides(image.shape)).buffer);

  const out = backend.makeOutput(outShape, image.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;

  const imageData = backend.dataIdMap.get(image.dataId);
  const imageId = imageData.id;

  const transformsData = backend.dataIdMap.get(transforms.dataId);
  const transformsId = transformsData.id;

  const interpolationModeId = interpolation === 'nearest' ? 1 : 2;
  let fillModeId;
  switch (fillMode) {
    case 'constant':
      fillModeId = 1;
      break;
    case 'reflect':
      fillModeId = 2;
      break;
    case 'wrap':
      fillModeId = 3;
      break;
    case 'nearest':
      fillModeId = 4;
      break;
    default:
      fillModeId = 1;
      break;
  }

  wasmTransform(
      imageId, transformsId, (transforms.shape[0] > 1), batch, outHeight,
      outWidth, numChannels, imageWidth, imageHeight, strides,
      image.shape.length - 1, interpolationModeId, fillModeId, fillValue,
      outId);

  return out;
}

export const transformConfig: KernelConfig = {
  kernelName: Transform,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: transform as {} as KernelFunc
};
