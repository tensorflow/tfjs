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
  images: TensorInfo;
  boxes: TensorInfo;
  boxInd: TensorInfo;
}

interface CropAndResizeAttrs extends NamedAttrMap {
  method: InterpolationMethod;
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
    imageStrides: Uint8Array, imageStridesLength: number,
    outputStrides: Uint8Array, outputStridesLength: number,
    imagesShape: Uint8Array, imagesShapeLength: number, cropSize: Uint8Array,
    cropSizeLength: number, method: number, extrapolationValue: number,
    outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmCropAndResize = backend.wasm.cwrap('CropAndResize', null /*void*/, [
    'number',  // imagesId
    'number',  // boxesId
    'number',  // boxIndId
    'number',  // numBoxes
    'array',   // image strides
    'number',  // image strides length
    'array',   // output strides
    'number',  // output strides length
    'array',   // images shape
    'number',  // images shape length
    'array',   // cropSize
    'number',  // cropSize length
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

  const imageStrides = util.computeStrides(images.shape);
  const outputStrides = util.computeStrides(outShape);

  const imageStridesBytes = new Uint8Array(new Int32Array(imageStrides).buffer);
  const outputStridesBytes =
      new Uint8Array(new Int32Array(outputStrides).buffer);
  const imagesShapeBytes = new Uint8Array(new Int32Array(images.shape).buffer);
  const cropSizeBytes =
      new Uint8Array(new Int32Array(cropSize as [number, number]).buffer);

  wasmCropAndResize(
      imagesId, boxesId, boxIndId, numBoxes, imageStridesBytes,
      imageStrides.length, outputStridesBytes, outputStrides.length,
      imagesShapeBytes, images.shape.length, cropSizeBytes,
      (cropSize as [number, number]).length,
      InterpolationMethod[method as keyof typeof InterpolationMethod],
      extrapolationValue as number, outId);
  return out;
}

registerKernel({
  kernelName: 'CropAndResize',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: cropAndResize
});
