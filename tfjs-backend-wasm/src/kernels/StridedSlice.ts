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

import {KernelConfig, KernelFunc, slice_util, StridedSlice, StridedSliceAttrs, StridedSliceInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {BackendWasm} from '../backend_wasm';
import {reshape} from './Reshape';
import {slice} from './Slice';

let wasmStridedSlice: (
    xId: number, xStridesBytes: Uint8Array, xRank: number,
    beginBytes: Uint8Array, endBytes: Uint8Array, stridesBytes: Uint8Array,
    outShapeBytes: Uint8Array, outStridesBytes: Uint8Array,
    outShapeLength: number, outId: number) => void;

function setup(backend: BackendWasm): void {
  wasmStridedSlice = backend.wasm.cwrap(StridedSlice, null /*void*/, [
    'number',  // xId
    'array',   // xStrides
    'number',  // xRank
    'array',   // beginBytes
    'array',   // endBytes
    'array',   // stridesBytes
    'array',   // outShapeBytes
    'array',   // outStridesBytes
    'number',  // outShapeLength
    'number',  // outId
  ]);
}

export function stridedSlice(args: {
  backend: BackendWasm,
  inputs: StridedSliceInputs,
  attrs: StridedSliceAttrs
}): TensorInfo {
  const {backend, inputs, attrs} = args;
  const {x} = inputs;

  const {
    begin,
    end,
    strides,
    beginMask,
    endMask,
    ellipsisMask,
    newAxisMask,
    shrinkAxisMask
  } = attrs;

  const {
    finalShapeSparse,
    finalShape,
    isIdentity,
    sliceDim0,
    isSimpleSlice,
    begin: $begin,
    end: $end,
    strides: $strides
  } =
      slice_util.sliceInfo(
          x.shape, begin, end, strides, beginMask, endMask, ellipsisMask,
          newAxisMask, shrinkAxisMask);

  let result;

  if (isIdentity) {
    // Optimization #1, slice is a no-op plus reshape
    result = reshape({inputs: {x}, backend, attrs: {shape: finalShape}});
  } else if (sliceDim0 || isSimpleSlice) {
    // Optimization #2, slice is memory contiguous (only occurs in dim 0)
    util.assert(
        x.shape.length >= 1,
        () => `Input must have rank at least 1, got: ${x.shape.length}`);

    const size = slice_util.computeOutShape($begin, $end, $strides);
    // To tolerate begin[0] > end[0] (a 0-output slice), we min(begin, end).
    const sliced = slice({inputs: {x}, backend, attrs: {begin: $begin, size}});
    result =
        reshape({inputs: {x: sliced}, backend, attrs: {shape: finalShape}});
    backend.disposeData(sliced.dataId);
  } else {
    const out = backend.makeOutput(finalShapeSparse, 'float32');

    const xId = backend.dataIdMap.get(x.dataId).id;
    const xStridesBytes =
        new Uint8Array(new Int32Array(util.computeStrides(x.shape)).buffer);
    const beginBytes = new Uint8Array(new Int32Array($begin).buffer);
    const endBytes = new Uint8Array(new Int32Array($end).buffer);
    const stridesBytes = new Uint8Array(new Int32Array($strides).buffer);

    const outputShapeBytes =
        new Uint8Array(new Int32Array(finalShapeSparse).buffer);
    const outStridesBytes = new Uint8Array(
        new Int32Array(util.computeStrides(finalShapeSparse)).buffer);
    const outId = backend.dataIdMap.get(out.dataId).id;

    wasmStridedSlice(
        xId, xStridesBytes, x.shape.length, beginBytes, endBytes, stridesBytes,
        outputShapeBytes, outStridesBytes, finalShapeSparse.length, outId);

    result = reshape({inputs: {x: out}, backend, attrs: {shape: finalShape}});

    backend.disposeData(out.dataId);
  }

  return result;
}

export const stridedSliceConfig: KernelConfig = {
  kernelName: StridedSlice,
  backendName: 'wasm',
  setupFunc: setup,
  kernelFunc: stridedSlice as {} as KernelFunc
};
