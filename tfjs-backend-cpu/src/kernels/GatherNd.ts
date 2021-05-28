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

import {backend_util, GatherNd, GatherNdInputs, KernelConfig, KernelFunc, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendCPU} from '../backend_cpu';

import {gatherNdImpl} from './GatherNd_Impl';

export function gatherNd(
    args: {inputs: GatherNdInputs, backend: MathBackendCPU}): TensorInfo {
  const {inputs, backend} = args;
  const {params, indices} = inputs;

  const paramsSize = util.sizeFromShape(params.shape);

  const indicesShape = indices.shape;
  const sliceRank = indicesShape[indicesShape.length - 1];

  const [resultShape, numSlices, sliceSize, strides] =
      backend_util.prepareAndValidate(params, indices);
  if (numSlices === 0) {
    return backend.makeTensorInfo(resultShape, params.dtype, []);
  }

  const indicesData = backend.data.get(indices.dataId).values as TypedArray;
  const paramsBuf = backend.bufferSync(params);
  const outBuf = gatherNdImpl(
      indicesData, paramsBuf, params.dtype, numSlices, sliceRank, sliceSize,
      strides, params.shape, paramsSize);

  return backend.makeTensorInfo(resultShape, params.dtype, outBuf.values);
}

export const gatherNdConfig: KernelConfig = {
  kernelName: GatherNd,
  backendName: 'cpu',
  kernelFunc: gatherNd as {} as KernelFunc
};
