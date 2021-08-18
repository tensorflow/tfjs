/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {KernelConfig, KernelFunc, PadV2, PadV2Attrs, PadV2Inputs, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {fill} from './Fill';

import {CppDType} from './types';

let wasmPadV2: (
    xId: number, xShapeBytes: Uint8Array, xShapeLength: number, xDtype: number,
    prePaddingsBytes: Uint8Array, postPaddingsBytes: Uint8Array,
    constantValue: number, outId: number) => void;

function setup(backend: BackendWasm) {
  wasmPadV2 = backend.wasm.cwrap(PadV2, null /* void */, [
    'number',  // xId
    'array',   // x.shape
    'number',  // x.shape.length
    'number',  // x.dtype
    'array',   // pre-paddings
    'array',   // post-paddings
    'number',  // constantValue
    'number',  // outId
  ]);
}

function pad(
    args: {inputs: PadV2Inputs, backend: BackendWasm, attrs: PadV2Attrs}) {
  const {inputs: {x}, backend, attrs: {paddings, constantValue}} = args;

  const outShape = paddings.map(
      (p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);

  if (util.sizeFromShape(x.shape) === 0) {
    // Short-circuit the computation, since x doesn't have value, only
    // the shape is used to compute output shape to pad.
    return fill({
      backend,
      attrs: {shape: outShape, value: constantValue, dtype: x.dtype}
    });
  }

  const xId = backend.dataIdMap.get(x.dataId).id;
  const out = backend.makeOutput(outShape, x.dtype);
  const outTensorData = backend.dataIdMap.get(out.dataId);
  const outId = outTensorData.id;

  const xShapeBytes = new Uint8Array(new Int32Array(x.shape).buffer);

  const prePaddingsFlat = paddings.map(padTuple => padTuple[0]);
  const postPaddingsFlat = paddings.map(padTuple => padTuple[1]);
  const prePaddingsBytes =
      new Uint8Array(new Int32Array(prePaddingsFlat).buffer);
  const postPaddingsBytes =
      new Uint8Array(new Int32Array(postPaddingsFlat).buffer);

  wasmPadV2(
      xId, xShapeBytes, x.shape.length, CppDType[x.dtype], prePaddingsBytes,
      postPaddingsBytes, constantValue, outId);
  return out;
}

export const padV2Config: KernelConfig = {
  kernelName: PadV2,
  backendName: 'wasm',
  kernelFunc: pad as {} as KernelFunc,
  setupFunc: setup
};
