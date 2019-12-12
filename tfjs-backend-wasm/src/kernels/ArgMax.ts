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

import {KernelFunc, registerKernel, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {CppDType} from './types';

interface ArgMaxInputs {
  x: TensorInfo;
}

interface ArgMaxAttrs {
  axis: number;
}

let wasmFunc: (
    xId: number, dtype: number, outerSize: number, innerSize: number,
    outId: number) => void;

function setup(backend: BackendWasm) {
  wasmFunc = backend.wasm.cwrap('ArgMax', null /* void */, [
    'number',  // x_id
    'number',  // dtype
    'number',  // outer_size
    'number',  // inner_size
    'number'   // out_id
  ]);
}

function argmax(
    args: {inputs: ArgMaxInputs, backend: BackendWasm, attrs: ArgMaxAttrs}) {
  const {inputs: {x}, backend, attrs: {axis}} = args;
  const outShape = x.shape.slice(0, -1);
  const out = backend.makeOutput(outShape, 'int32');
  const xId = backend.dataIdMap.get(x.dataId).id;
  const outId = backend.dataIdMap.get(out.dataId).id;
  const outerSize = util.sizeFromShape(out.shape);
  const innerSize = x.shape[axis];
  wasmFunc(xId, CppDType[x.dtype], outerSize, innerSize, outId);
  return out;
}

registerKernel({
  kernelName: 'ArgMax',
  backendName: 'wasm',
  kernelFunc: argmax as {} as KernelFunc,
  setupFunc: setup
});
