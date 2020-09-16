/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {KernelConfig, KernelFunc, NumericDataType, Slice, slice_util, SliceAttrs, SliceInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';

export function slice(
    args: {inputs: SliceInputs, backend: MathBackendCPU, attrs: SliceAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {begin, size} = attrs;

  assertNotComplex(x, 'slice');

  const xRank = x.shape.length;
  const xStrides = util.computeStrides(x.shape);

  const [$begin, $size] = slice_util.parseSliceParams(x, begin, size);
  slice_util.assertParamsValid(x, $begin, $size);

  const isContinous = slice_util.isSliceContinous(x.shape, $begin, $size);
  const vals = backend.data.get(x.dataId).values as TypedArray;
  const length = util.sizeFromShape($size);

  if (isContinous) {
    const flatOffset = slice_util.computeFlatOffset($begin, xStrides);
    const resultVals = vals.subarray(flatOffset, flatOffset + length);
    return backend.makeTensorInfo($size, x.dtype, resultVals);
  }

  const outVals =
      util.getTypedArrayFromDType(x.dtype as NumericDataType, length);

  for (let i = 0; i < length; ++i) {
    const rank = $size.length;
    const strides = util.computeStrides($size);
    const loc = util.indexToLoc(i, rank, strides);
    const xLoc = loc.map((idx: number, j) => idx + $begin[j]);
    const xIndex = util.locToIndex(xLoc, xRank, xStrides);
    outVals[i] = vals[xIndex];
  }

  return backend.makeTensorInfo($size, x.dtype, outVals);
}

export const sliceConfig: KernelConfig = {
  kernelName: Slice,
  backendName: 'cpu',
  kernelFunc: slice as {} as KernelFunc
};
