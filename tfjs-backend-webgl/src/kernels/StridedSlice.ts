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

import {buffer, KernelConfig, KernelFunc, Rank, slice_util, StridedSlice, StridedSliceAttrs, StridedSliceInputs, TensorBuffer, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

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

  const {nonStrided, $begin, $strides, size, newShape, outShape} =
      slice_util.sliceInfo(
          x.shape, begin, end, strides, beginMask, endMask, ellipsisMask,
          newAxisMask, shrinkAxisMask);

  const $x = reshape({inputs: {x}, backend, attrs: {shape: newShape}});

  let result;
  if (nonStrided) {
    const sliced =
        slice({inputs: {x: $x}, backend, attrs: {begin: $begin, size}});
    result = reshape({inputs: {x: sliced}, backend, attrs: {shape: outShape}});

    backend.disposeIntermediateTensorInfo(sliced);
  } else if (outShape.some(axis => axis === 0)) {
    result = backend.makeTensorInfo(outShape, x.dtype, []);
  } else {
    const shouldExecuteOnCPU = backend.shouldExecuteOnCPU([$x]);
    if (shouldExecuteOnCPU) {
      const xTexData = backend.texData.get($x.dataId);
      const values = xTexData.values as TypedArray;
      const xBuf = buffer($x.shape, $x.dtype, values) as TensorBuffer<Rank>;
      const resultValues =
          stridedSliceImplCPU(outShape, xBuf, $strides, $begin);
      result = backend.makeTensorInfo(outShape, $x.dtype, resultValues.values);
    } else {
      const program = new StridedSliceProgram($begin, $strides, outShape);
      result = backend.runWebGLProgram(program, [$x], $x.dtype);
    }
  }

  const resultReshaped =
      reshape({inputs: {x: result}, backend, attrs: {shape: outShape}});

  backend.disposeIntermediateTensorInfo($x);
  backend.disposeIntermediateTensorInfo(result);

  return resultReshaped;
}

export const stridedSliceConfig: KernelConfig = {
  kernelName: StridedSlice,
  backendName: 'webgl',
  kernelFunc: stridedSlice as {} as KernelFunc
};
