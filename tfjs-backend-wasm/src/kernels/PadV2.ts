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
import {CppDType} from './types';

interface PadInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface PadAttrs extends NamedAttrMap {
  paddings: Array<[number, number]>;
  constantValue: number;
}

let wasmPadV2: (
    xId: number, xShapeBytes: Uint8Array, xShapeLength: number, xDtype: number,
    paddingsBytes: Uint8Array, constantValue: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmPadV2 = backend.wasm.cwrap('PadV2', null /* void */, [
    'number',  // xId
    'array',   // x.shape
    'number',  // x.shape.length
    'number',  // x.dtype
    'array',   // paddings
    'number',  // constantValue
    'number',  // outId
  ]);
}

function pad(args: {inputs: PadInputs, backend: BackendWasm, attrs: PadAttrs}) {
  const {inputs: {x}, backend, attrs: {paddings, constantValue}} = args;

  const outShape = paddings.map(
      (p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
  const xId = backend.dataIdMap.get(x.dataId).id;
  const out = backend.makeOutput(outShape, x.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;
  const xShapeBytes = new Uint8Array(new Int32Array(x.shape).buffer);
  const paddingsFlat = [].concat(...paddings);
  const paddingsBytes = new Uint8Array(new Int32Array(paddingsFlat).buffer);
  wasmPadV2(
      xId, xShapeBytes, x.shape.length, CppDType[x.dtype], paddingsBytes,
      constantValue, outId);
  return out;
}

registerKernel({
  kernelName: 'PadV2',
  backendName: 'wasm',
  kernelFunc: pad,
  setupFunc: setup
});
