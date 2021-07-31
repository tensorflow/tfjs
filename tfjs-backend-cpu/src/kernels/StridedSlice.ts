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

import {KernelConfig, KernelFunc, slice_util, StridedSlice, StridedSliceAttrs, StridedSliceInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';
import {assertNotComplex} from '../cpu_util';
import {reshape} from './Reshape';
import {slice} from './Slice';
import {stridedSliceImpl} from './StridedSlice_impl';

export function stridedSlice(args: {
  inputs: StridedSliceInputs,
  backend: MathBackendCPU,
  attrs: StridedSliceAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
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

  assertNotComplex(x, 'stridedSlice');

  const {
    finalShape,
    isIdentity,
    sliceDim0,
    begin: $begin,
    end: $end,
    strides: $strides
  } =
      slice_util.sliceInfo(
          x.shape, begin, end, strides, beginMask, endMask, ellipsisMask,
          newAxisMask, shrinkAxisMask);

  console.log(`isIdentity: ${isIdentity}`);
  console.log(`sliceDim0: ${sliceDim0}`);
  console.log($begin);
  console.log($end);
  console.log($strides);

  let result;

  if (isIdentity) {
    // Optimization #1, slice is a no-op plus reshape
    result = reshape({inputs: {x}, backend, attrs: {shape: finalShape}});
  } else if (sliceDim0) {
    // Optimization #2, slice is memory contiguous (only occurs in dim 0)
    util.assert(
        x.shape.length >= 1,
        () => `Input must have rank at least 1, got: ${x.shape.length}`);

    const size = slice_util.computeOutShape($begin, $end, $strides);
    // To tolerate begin[0] > end[0] (a 0-output slice), we min(begin, end).
    const sliced = slice({inputs: {x}, backend, attrs: {begin: $begin, size}});
    result =
        reshape({inputs: {x: sliced}, backend, attrs: {shape: finalShape}});
    backend.disposeIntermediateTensorInfo(sliced);
  } else {
    const xBuf = backend.bufferSync(x);
    console.log('normal path');
    console.log($begin);
    console.log($strides);
    console.log(xBuf);
    const outBuf = stridedSliceImpl(finalShape, xBuf, $strides, $begin);

    result = backend.makeTensorInfo(outBuf.shape, outBuf.dtype, outBuf.values);
  }

  const resultReshaped =
      reshape({inputs: {x: result}, backend, attrs: {shape: finalShape}});

  backend.disposeIntermediateTensorInfo(result);

  return resultReshaped;
}

export const stridedSliceConfig: KernelConfig = {
  kernelName: StridedSlice,
  backendName: 'cpu',
  kernelFunc: stridedSlice as {} as KernelFunc
};
