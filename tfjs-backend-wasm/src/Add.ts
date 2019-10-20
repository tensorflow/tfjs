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

import {backend_util, registerKernel, TensorInfo, util} from '@tensorflow/tfjs-core';
import {BackendWasm} from './backend_wasm';

interface AddInputs {
  a: TensorInfo;
  b: TensorInfo;
}

let wasmAdd: (aId: number, bId: number, outId: number) => void;

registerKernel('Add', 'wasm', ({inputs, storage}) => {
  const backend = storage as BackendWasm;
  if (wasmAdd == null) {
    wasmAdd = backend.wasm.cwrap(
        'add', null /* void */, ['number', 'number', 'number']);
  }
  const {a, b} = inputs as {} as AddInputs;
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
});
