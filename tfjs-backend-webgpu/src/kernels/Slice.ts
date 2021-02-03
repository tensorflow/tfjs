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

import {KernelConfig, KernelFunc, Slice, slice_util, SliceAttrs, SliceInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {sliceImplCPU} from '../kernel_utils/shared';
import {SliceProgram} from './slice_webgpu';

export function slice(
    args: {inputs: SliceInputs, backend: WebGPUBackend, attrs: SliceAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {begin, size} = attrs;

  const [$begin, $size] = slice_util.parseSliceParams(x, begin, size);
  slice_util.assertParamsValid(x, $begin, $size);

  if (backend.shouldExecuteOnCPU([x]) || x.dtype === 'string') {
    const xBufferInfo = backend.tensorMap.get(x.dataId);
    const outValues = sliceImplCPU(
        xBufferInfo.values as TypedArray, $begin, $size, x.shape, x.dtype);
    return backend.makeTensorInfo($size, x.dtype, outValues);
  }

  if (util.sizeFromShape($size) === 0) {
    return backend.makeTensorInfo($size, x.dtype, []);
  }

  // TODO(xing.xu): Add shadow slice support.
  const program = new SliceProgram($begin, $size);
  return backend.runWebGPUProgram(program, [x], x.dtype);
}

export const sliceConfig: KernelConfig = {
  kernelName: Slice,
  backendName: 'webgpu',
  kernelFunc: slice as {} as KernelFunc
};
