/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {KernelConfig, KernelFunc, MirrorPad, MirrorPadAttrs, MirrorPadInputs} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

import {CppDType} from './types';

// Must match enum in MirrorPad.cc
enum MirrorPaddingMode {
  reflect = 0,
  symmetric = 1
}

let wasmMirrorPad: (
    xId: number, xShapeBytes: Uint8Array, xShapeLength: number, xDtype: number,
    prePaddingsBytes: Uint8Array, postPaddingsBytes: Uint8Array, mode: number,
    outId: number) => void;

function setup(backend: BackendWasm) {
  wasmMirrorPad = backend.wasm.cwrap(MirrorPad, null /* void */, [
    'number',  // xId
    'array',   // x.shape
    'number',  // x.shape.length
    'number',  // x.dtype
    'array',   // pre-paddings
    'array',   // post-paddings
    'number',  // mode
    'number',  // outId
  ]);
}

function mirrorPad(args: {
  inputs: MirrorPadInputs,
  backend: BackendWasm,
  attrs: MirrorPadAttrs
}) {
  const {inputs: {x}, backend, attrs: {paddings, mode}} = args;

  const outShape = paddings.map(
      (p, i) => p[0] /* beforePad */ + x.shape[i] + p[1] /* afterPad */);
  const xId = backend.dataIdMap.get(x.dataId).id;
  const out = backend.makeOutput(outShape, x.dtype);
  const outId = backend.dataIdMap.get(out.dataId).id;
  const xShapeBytes = new Uint8Array(new Int32Array(x.shape).buffer);

  const prePaddingsFlat = paddings.map(padTuple => padTuple[0]);
  const postPaddingsFlat = paddings.map(padTuple => padTuple[1]);
  const prePaddingsBytes =
      new Uint8Array(new Int32Array(prePaddingsFlat).buffer);
  const postPaddingsBytes =
      new Uint8Array(new Int32Array(postPaddingsFlat).buffer);

  wasmMirrorPad(
      xId, xShapeBytes, x.shape.length, CppDType[x.dtype], prePaddingsBytes,
      postPaddingsBytes, MirrorPaddingMode[mode], outId);
  return out;
}

export const mirrorPadConfig: KernelConfig = {
  kernelName: MirrorPad,
  backendName: 'wasm',
  kernelFunc: mirrorPad as {} as KernelFunc,
  setupFunc: setup
};
