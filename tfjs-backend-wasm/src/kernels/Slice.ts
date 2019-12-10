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

import {backend_util, buffer, NamedAttrMap, NamedTensorInfoMap, registerKernel, slice_util, util} from '@tensorflow/tfjs-core';
import {TensorInfo} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';

interface SliceInputs extends NamedTensorInfoMap {
  x: TensorInfo;
}

interface SliceAttrs extends NamedAttrMap {
  begin: number[];
  size: number[];
}

export function slice(
    args: {inputs: SliceInputs, attrs: SliceAttrs, backend: BackendWasm}) {
  const {inputs: {x}, attrs: {begin, size}, backend} = args;
  const isContinous = slice_util.isSliceContinous(x.shape, begin, size);
  const xVals = backend.typedArrayFromHeap(x);
  const out = backend.makeOutput(size, x.dtype);
  const outVals = backend.typedArrayFromHeap(out);
  const xStrides = util.computeStrides(x.shape);
  if (isContinous) {
    const flatOffset = slice_util.computeFlatOffset(begin, xStrides);
    outVals.set(
        xVals.subarray(flatOffset, flatOffset + util.sizeFromShape(size)));
    return out;
  }
  const rank = x.shape.length;
  if (rank === 2) {
    slice2d(
        xVals, xStrides[0], outVals, begin as [number, number],
        size as [number, number]);
  } else if (rank === 3) {
    slice3d(
        xVals, xStrides[0], xStrides[1], outVals,
        begin as [number, number, number], size as [number, number, number]);
  } else if (rank === 4) {
    slice4d(
        xVals, xStrides[0], xStrides[1], xStrides[2], outVals,
        begin as [number, number, number, number],
        size as [number, number, number, number]);
  } else {
    genericSliceSlow(xVals, x, outVals, begin, size);
  }
  return out;
}

function slice2d(
    xVals: backend_util.TypedArray, xStride: number,
    outVals: backend_util.TypedArray, begin: [number, number],
    size: [number, number]): void {
  let outOffset = 0;
  const beginI = begin[0];
  const beginJ = begin[1];
  const endI = beginI + size[0];
  for (let i = beginI; i < endI; i++) {
    const xOffset = i * xStride + beginJ;
    outVals.set(xVals.subarray(xOffset, xOffset + size[1]), outOffset);
    outOffset += size[1];
  }
}

function slice3d(
    xVals: backend_util.TypedArray, xStride1: number, xStride2: number,
    outVals: backend_util.TypedArray, begin: [number, number, number],
    size: [number, number, number]): void {
  let outOffset = 0;
  const beginI = begin[0];
  const beginJ = begin[1];
  const beginK = begin[2];
  const endI = beginI + size[0];
  const endJ = beginJ + size[1];
  for (let i = beginI; i < endI; i++) {
    for (let j = beginJ; j < endJ; j++) {
      const xOffset = i * xStride1 + j * xStride2 + beginK;
      outVals.set(xVals.subarray(xOffset, xOffset + size[2]), outOffset);
      outOffset += size[2];
    }
  }
}

function slice4d(
    xVals: backend_util.TypedArray, xStride1: number, xStride2: number,
    xStride3: number, outVals: backend_util.TypedArray,
    begin: [number, number, number, number],
    size: [number, number, number, number]): void {
  let outOffset = 0;
  const beginI = begin[0];
  const beginJ = begin[1];
  const beginK = begin[2];
  const endI = beginI + size[0];
  const endJ = beginJ + size[1];
  const endK = beginK + size[2];
  const beginL = begin[3];

  for (let i = beginI; i < endI; i++) {
    for (let j = beginJ; j < endJ; j++) {
      for (let k = beginK; k < endK; k++) {
        const xOffset = i * xStride1 + j * xStride2 + k * xStride3 + beginL;
        outVals.set(xVals.subarray(xOffset, xOffset + size[3]), outOffset);
        outOffset += size[3];
      }
    }
  }
}

function genericSliceSlow(
    xVals: backend_util.TypedArray, xInfo: TensorInfo,
    outVals: backend_util.TypedArray, begin: number[], size: number[]): void {
  const outBuf = buffer(size, xInfo.dtype, outVals);
  const xBuf = buffer(xInfo.shape, xInfo.dtype, xVals);
  for (let i = 0; i < outBuf.size; ++i) {
    const loc = outBuf.indexToLoc(i);
    const xLoc = loc.map((idx, j) => idx + begin[j]);
    outVals[i] = xBuf.get(...xLoc) as number;
  }
}

registerKernel({
  kernelName: 'Slice',
  backendName: 'wasm',
  kernelFunc: slice,
});
