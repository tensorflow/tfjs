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
  x: TensorInfo;
}

interface CropAndResizeAttrs extends NamedAttrMap {
  axes: number[];
}

let wasmCropAndResize: (outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmCropAndResize = backend.wasm.cwrap(
      'CropAndResize', null /*void*/, ['number, number, number']);
}

function cropAndResize(args: {
  backend: BackendWasm,
  inputs: CropAndResizeInputs,
  attrs: CropAndResizeAttrs
}): TensorInfo {
  const {backend, inputs, attrs} = args;
  const {method, extrapolationValue, cropSize} = attrs;
  const {images, boxes} = inputs;

  console.log('in cropandresize');
  console.log(inputs);
  console.log(method, extrapolationValue);
  const [batch, imageHeight, imageWidth, numChannels] = images.shape;
  const numBoxes = boxes.shape[0];
  console.log(batch, imageHeight, imageWidth);

  const [cropHeight, cropWidth] = cropSize as [number, number];
  // const xId = backend.dataIdMap.get(x.dataId).id;

  // backend_util.assertAxesAreInnerMostDims('max', axes, x.shape.length);
  // const [outShape, reduceShape] =
  //     backend_util.computeOutAndReduceShapes(x.shape, axes);
  // const reduceSize = util.sizeFromShape(reduceShape);
  const outShape = [numBoxes, cropHeight, cropWidth, numChannels];

  const out = backend.makeOutput(outShape, images.dtype);

  const outId = backend.dataIdMap.get(out.dataId).id;

  wasmCropAndResize(outId);
  return out;
}

registerKernel({
  kernelName: 'CropAndResize',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: cropAndResize
});
