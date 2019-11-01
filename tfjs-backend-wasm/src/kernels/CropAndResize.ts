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

import {NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface CropAndResizeInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface CropAndResizeAttrs extends NamedAttrMap {
  axes: number[];
}

let wasmCropAndResize: (
    imagesId: number, boxesId: number, boxIndId: number, numBoxes: number,
    imageStrides: [number, number, number],
    outputStrides: [number, number, number],
    batch: [number, number, number, number], cropSize: [number, number],
    method: number, extrapolationValue: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmCropAndResize = backend.wasm.cwrap('CropAndResize', null /*void*/, [
    'number',  // imagesId
    'number',  // boxesId
    'number',  // boxIndId
    'number',  // numBoxes
    'array',   // image strides
    'array',   // output strides
    'array',   // images shape
    'array',   // cropSize
    'number',  // method
    'number',  // extrapolation value
    'number'   // out id
  ]);
}

function cropAndResize(args: {
  backend: BackendWasm,
  inputs: CropAndResizeInputs,
  attrs: CropAndResizeAttrs
}): TensorInfo {
  const {backend, inputs, attrs} = args;
  const {method, extrapolationValue, cropSize} = attrs;
  const {images, boxes, boxInd} = inputs;

  const numBoxes = boxes.shape[0];

  const [cropHeight, cropWidth] = cropSize as [number, number];
  const outShape = [numBoxes, cropHeight, cropWidth, images.shape[3]];

  const imagesId = backend.dataIdMap.get(images.dataId).id;
  const boxesId = backend.dataIdMap.get(boxes.dataId).id;
  const boxIndId = backend.dataIdMap.get(boxInd.dataId).id;

  const out = backend.makeOutput(outShape, images.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;

  wasmCropAndResize(
      imagesId, boxesId, boxIndId, numBoxes,
      util.computeStrides(images.shape) as [number, number, number],
      util.computeStrides(outShape) as [number, number, number],
      images.shape as [number, number, number, number],
      cropSize as [number, number], method === 'bilinear' ? 1 : 0,
      extrapolationValue as number, outId);
  return out;
}

registerKernel({
  kernelName: 'CropAndResize',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: cropAndResize
});
