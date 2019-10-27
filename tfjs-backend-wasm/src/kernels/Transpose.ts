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

import {backend_util, buffer, NamedAttrMap, NamedTensorInfoMap, registerKernel, TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface TransposeInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface TransposeAttrs extends NamedAttrMap {
  perm: number[];
}

function transpose(
    args:
        {inputs: TransposeInputs, backend: BackendWasm, attrs: TransposeAttrs}):
    TensorInfo {
  const {inputs: {x}, backend, attrs: {perm}} = args;
  const newShape: number[] = new Array(x.shape.length);
  for (let i = 0; i < newShape.length; i++) {
    newShape[i] = x.shape[perm[i]];
  }
  const xVals = backend.typedArrayFromHeap(x);
  const out = backend.makeOutput(newShape, x.dtype);
  const outVals = backend.typedArrayFromHeap(out);
  genericSlowTranspose(xVals, x, outVals, out, perm);
  return out;
}

function genericSlowTranspose(
    xVals: backend_util.TypedArray, xInfo: TensorInfo,
    outVals: backend_util.TypedArray, outInfo: TensorInfo,
    perm: number[]): void {
  const xBuf = buffer(xInfo.shape, xInfo.dtype, xVals);
  const outBuf = buffer(outInfo.shape, outInfo.dtype, outVals);
  for (let i = 0; i < xBuf.size; ++i) {
    const loc = xBuf.indexToLoc(i);
    // Permute location.
    const newLoc: number[] = new Array(loc.length);
    for (let i = 0; i < newLoc.length; i++) {
      newLoc[i] = loc[perm[i]];
    }
    const newIndex = outBuf.locToIndex(newLoc);
    outVals[newIndex] = xVals[i];
  }
}

registerKernel({
  kernelName: 'Transpose',
  backendName: 'wasm',
  kernelFunc: transpose,
});
