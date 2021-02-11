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

import {buffer, KernelConfig, KernelFunc, Rank, slice_util, StridedSlice, StridedSliceAttrs, StridedSliceInputs, TensorBuffer, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {stridedSliceImplCPU} from '../kernel_utils/shared';

import {reshape} from './Reshape';
import {slice} from './Slice';
import {StridedSliceProgram} from './strided_slice_webgpu';

export function stridedSlice(args: {
  inputs: StridedSliceInputs,
  backend: WebGPUBackend,
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
    backend.disposeData(sliced.dataId);

  } else if (outShape.some(axis => axis === 0)) {
    result = backend.makeTensorInfo(outShape, x.dtype, []);
  } else {
    const shouldExecuteOnCPU = backend.shouldExecuteOnCPU([$x]);
    if (shouldExecuteOnCPU) {
      const xBufferInfo = backend.tensorMap.get($x.dataId);
      const values = xBufferInfo.values as TypedArray;
      const xBuf = buffer($x.shape, $x.dtype, values) as TensorBuffer<Rank>;
      const resultValues =
          stridedSliceImplCPU(outShape, xBuf, $strides, $begin);
      result = backend.makeTensorInfo(outShape, $x.dtype, resultValues.values);
    } else {
      const program = new StridedSliceProgram($begin, $strides, outShape);
      result = backend.runWebGPUProgram(program, [$x], $x.dtype);
    }
  }

  const resultReshaped =
      reshape({inputs: {x: result}, backend, attrs: {shape: outShape}});

  backend.disposeData($x.dataId);
  backend.disposeData(result.dataId);

  return resultReshaped;
}

export const stridedSliceConfig: KernelConfig = {
  kernelName: StridedSlice,
  backendName: 'webgpu',
  kernelFunc: stridedSlice as {} as KernelFunc
};
