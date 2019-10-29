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

import {backend_util, NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface MaxInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface MaxAttrs extends NamedAttrMap {
  axes: number[];
}

let wasmMax: (xId: number, reduceSize: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmMax =
      backend.wasm.cwrap('Max', null /*void*/, ['number, number, number']);
}

function max(args: {backend: BackendWasm, inputs: MaxInputs, attrs: MaxAttrs}):
    TensorInfo {
  const {backend, inputs, attrs} = args;
  const {axes} = attrs;
  const {x} = inputs;
  const xId = backend.dataIdMap.get(x.dataId).id;

  backend_util.assertAxesAreInnerMostDims('max', axes, x.shape.length);
  const [outShape, reduceShape] =
      backend_util.computeOutAndReduceShapes(x.shape, axes);
  const reduceSize = util.sizeFromShape(reduceShape);

  const out = backend.makeOutput(outShape, x.dtype);
  if (util.sizeFromShape(x.shape) === 0) {
    return out;
  }

  const outId = backend.dataIdMap.get(out.dataId).id;

  wasmMax(xId, reduceSize, outId);
  return out;
}

registerKernel({
  kernelName: 'Max',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: max
});
