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

interface CropAndResizeInputs extends NamedTensorInfoMap {
  images: TensorInfo;
  boxes: TensorInfo;
  boxInd: TensorInfo;
}

interface CropAndResizeAttrs extends NamedAttrMap {
  method: keyof InterpolationMethod;
  extrapolationValue: number;
  cropSize: [number, number];
}

// Must match enum in CropAndResize.cc
enum InterpolationMethod {
  bilinear = 0,
  nearest = 1
}

let wasmCropAndResize: (
    imagesId: number, boxesId: number, boxIndId: number, numBoxes: number,
    imagesShape: Uint8Array, imagesShapeLength: number, cropHeight: number,
    cropWidth: number, method: number, extrapolationValue: number,
    outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmCropAndResize = backend.wasm.cwrap('CropAndResize', null /*void*/, [
    'number',  // imagesId
    'number',  // boxesId
    'number',  // boxIndId
    'number',  // numBoxes
    'array',   // images shape
    'number',  // images shape length
    'number',  // cropHeight
    'number',  // cropWidth
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

  const imagesShapeBytes = new Uint8Array(new Int32Array(images.shape).buffer);

  wasmCropAndResize(
      imagesId, boxesId, boxIndId, numBoxes, imagesShapeBytes,
      images.shape.length, cropHeight, cropWidth,
      InterpolationMethod[method as {} as keyof typeof InterpolationMethod],
      extrapolationValue, outId);
  return out;
}

registerKernel({
  kernelName: 'CropAndResize',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: cropAndResize
});
