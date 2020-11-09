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

import {env, KernelConfig, KernelFunc, Slice, slice_util, SliceAttrs, SliceInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {sliceImplCPU} from '../kernel_utils/shared';
import {SliceProgram} from '../slice_gpu';
import {SlicePackedProgram} from '../slice_packed_gpu';

function shallowSlice(
    x: TensorInfo, begin: number[], size: number[], backend: MathBackendWebGL) {
  const xTexData = backend.texData.get(x.dataId);
  const t = backend.makeTensorInfo(size, x.dtype);
  const newTexData = backend.texData.get(t.dataId);
  // Copy texture data from the original tensor.
  Object.assign(newTexData, xTexData);
  newTexData.shape = size;
  newTexData.dtype = x.dtype;
  let flatOffset =
      slice_util.computeFlatOffset(begin, util.computeStrides(x.shape));
  if (xTexData.slice) {
    // We are slicing an already sliced tensor, so we have to accumulate
    // the offset.
    flatOffset += xTexData.slice.flatOffset;
  }
  newTexData.slice = {
    flatOffset,
    // Point to the original dataId, which is used to do ref counting.
    origDataId: xTexData.slice && xTexData.slice.origDataId || x.dataId
  };

  // Increase the ref count for that data bucket.
  const refCount = backend.dataRefCount.get(newTexData.slice.origDataId) || 1;
  backend.dataRefCount.set(newTexData.slice.origDataId, refCount + 1);
  return t;
}

export function slice(
    args: {inputs: SliceInputs, backend: MathBackendWebGL, attrs: SliceAttrs}):
    TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {begin, size} = attrs;

  const [$begin, $size] = slice_util.parseSliceParams(x, begin, size);
  slice_util.assertParamsValid(x, $begin, $size);

  if (util.sizeFromShape($size) === 0) {
    return backend.makeTensorInfo($size, x.dtype, []);
  }

  // Run on cpu if dtype is string. For string, the backend represents it
  // as Uint8Array[], where each Uint8Array is a character. Given that the
  // computation is only on the outer array, uploading the whole data onto
  // gpu is wasteful. Also, currently webgl doesn't have a design to
  // upload and retrieve Uint8Array[] between cpu and gpu. Therefore, we
  // just run the kernel on cpu if dtype is string.
  if (backend.shouldExecuteOnCPU([x]) || x.dtype === 'string') {
    const xTexData = backend.texData.get(x.dataId);
    const outValues = sliceImplCPU(
        xTexData.values as TypedArray, $begin, $size, x.shape, x.dtype);
    return backend.makeTensorInfo($size, x.dtype, outValues);
  }

  const {isPacked} = backend.texData.get(x.dataId);
  const isContinous = slice_util.isSliceContinous(x.shape, $begin, $size);
  if (isPacked || !isContinous) {
    const program = env().getBool('WEBGL_PACK_ARRAY_OPERATIONS') ?
        new SlicePackedProgram($size) :
        new SliceProgram($size);
    const customSetup = program.getCustomSetupFunc($begin);
    return backend.runWebGLProgram(program, [x], x.dtype, customSetup);
  }
  backend.uploadToGPU(x.dataId);
  return shallowSlice(x, $begin, $size, backend);
}

export const sliceConfig: KernelConfig = {
  kernelName: Slice,
  backendName: 'webgl',
  kernelFunc: slice as {} as KernelFunc
};
