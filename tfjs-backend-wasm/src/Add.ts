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

import {backend_util, NamedTensorInfoMap, registerKernel, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from './backend_wasm';

interface AddInputs extends NamedTensorInfoMap {
  a: TensorInfo;
  b: TensorInfo;
}

let wasmAdd: (aId: number, bId: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmAdd = backend.wasm.cwrap(
      'Add', null /* void */, ['number', 'number', 'number']);
}

function add(args: {backend: BackendWasm, inputs: AddInputs}): TensorInfo {
  const {backend, inputs} = args;
  const {a, b} = inputs;
  const aId = backend.dataIdMap.get(a.dataId).id;
  const bId = backend.dataIdMap.get(b.dataId).id;

  const newShape = backend_util.assertAndGetBroadcastShape(a.shape, b.shape);
  const out = backend.makeOutput(newShape, a.dtype);
  // Short-circuit zero-sized tensors.
  if (util.sizeFromShape(newShape) === 0) {
    return out;
  }

  const aBroadcastDims = backend_util.getBroadcastDims(a.shape, newShape);
  const bBroadcastDims = backend_util.getBroadcastDims(b.shape, newShape);
  const loopsOverAllOfA = aBroadcastDims.every((v, i) => v === i);
  const loopsOverAllOfB = bBroadcastDims.every((v, i) => v === i);
  const outId = backend.dataIdMap.get(out.dataId).id;

  if (loopsOverAllOfA && loopsOverAllOfB) {
    wasmAdd(aId, bId, outId);
    return out;
  } else {
    throw new Error('Broadcasting along inner dims is not yet supported');
  }
}

registerKernel({
  kernelName: 'Add',
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: add
});
