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

import {backend_util, KernelFunc, registerKernel, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

let wasmFunc: (aId: number, bId: number, dtype: number, outId: number) => void;

function setupFunc(backend: BackendWasm): void {
  wasmFunc = backend.wasm.cwrap('AddN', null /* void */, [
    'number',  // a_id,
    'number',  // b_id
    'number',  // dtype
    'number'   // out_id
  ]);

  function addn(args: {inputs: TensorInfo[], backend: BackendWasm}) {
    const {inputs, backend} = args;
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
      wasmFunc(aId, bId, CppDType[a.dtype], outId);
      return out;
    } else {
      throw new Error('Broadcasting along inner dims is not yet supported');
    }
  }

  registerKernel({
    kernelName: 'AddN',
    backendName: 'wasm',
    setupFunc,
    kernelFunc: addn as {} as KernelFunc,
  });
