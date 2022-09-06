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

import {buffer, KernelConfig, KernelFunc, Rank, slice_util, StridedSlice, StridedSliceAttrs, StridedSliceInputs, TensorBuffer, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {stridedSliceImplCPU} from '../kernel_utils/shared';
import {StridedSliceProgram} from '../strided_slice_gpu';

import {reshape} from './Reshape';
import {slice} from './Slice';

export function stridedSlice(args: {
  inputs: StridedSliceInputs,
  backend: MathBackendWebGL,
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
    backend.disposeIntermediateTensorInfo(sliced);
  } else {
    const shouldExecuteOnCPU = backend.shouldExecuteOnCPU([x]);
    if (shouldExecuteOnCPU) {
      // tslint:disable-next-line: no-unnecessary-type-assertion
      const values = backend.readSync(x.dataId) as TypedArray;
      // tslint:disable-next-line: no-unnecessary-type-assertion
      const xBuf = buffer(x.shape, x.dtype, values) as TensorBuffer<Rank>;
      const resultValues =
          stridedSliceImplCPU(finalShapeSparse, xBuf, $strides, $begin);
      result = backend.makeTensorInfo(finalShape, x.dtype, resultValues.values);
    } else {
      const program =
          new StridedSliceProgram($begin, $strides, finalShapeSparse);
      result = backend.runWebGLProgram(program, [x], x.dtype);
    }
  }

  const resultReshaped =
      reshape({inputs: {x: result}, backend, attrs: {shape: finalShape}});

  backend.disposeIntermediateTensorInfo(result);

  return resultReshaped;
}

export const stridedSliceConfig: KernelConfig = {
  kernelName: StridedSlice,
  backendName: 'webgl',
  kernelFunc: stridedSlice as {} as KernelFunc
};
