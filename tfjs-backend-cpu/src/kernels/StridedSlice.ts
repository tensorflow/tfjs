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

import {KernelConfig, KernelFunc, slice_util, StridedSlice, StridedSliceAttrs, StridedSliceInputs, TensorInfo} from '@tensorflow/tfjs-core';

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
    const xBuf = backend.bufferSync($x);
    const outBuf = stridedSliceImpl(outShape, xBuf, $strides, $begin);

    result = backend.makeTensorInfo(outBuf.shape, outBuf.dtype, outBuf.values);
  }

  const resultReshaped =
      reshape({inputs: {x: result}, backend, attrs: {shape: outShape}});

  backend.disposeIntermediateTensorInfo($x);
  backend.disposeIntermediateTensorInfo(result);

  return resultReshaped;
}

export const stridedSliceConfig: KernelConfig = {
  kernelName: StridedSlice,
  backendName: 'cpu',
  kernelFunc: stridedSlice as {} as KernelFunc
};
