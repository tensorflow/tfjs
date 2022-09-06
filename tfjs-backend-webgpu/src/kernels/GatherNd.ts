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

import {backend_util, GatherNd, GatherNdInputs, KernelConfig, KernelFunc, Rank, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {GatherNDProgram} from '../gather_nd_webgpu';
import {gatherNdImplCPU} from '../kernel_utils/shared';

import {reshape} from './Reshape';

export function gatherNd(
    args: {inputs: GatherNdInputs, backend: WebGPUBackend}): TensorInfo {
  const {inputs, backend} = args;
  const {params, indices} = inputs;

  const indicesShape = indices.shape;
  const sliceRank = indicesShape[indicesShape.length - 1];
  const paramsSize = util.sizeFromShape(params.shape);

  const [resultShape, numSlices, sliceSize, strides] =
      backend_util.prepareAndValidate(params, indices);

  const flattenIndices = reshape(
      {inputs: {x: indices}, backend, attrs: {shape: [numSlices, sliceRank]}});
  const flattenX = reshape({
    inputs: {x: params},
    backend,
    attrs: {shape: [(util.sizeFromShape(params.shape) / sliceSize), sliceSize]}
  });
  if (backend.shouldExecuteOnCPU([params, indices]) ||
      params.dtype === 'string') {
    const indicesData = backend.readSync(indices.dataId) as TypedArray;
    const paramsBuf = backend.bufferSync<Rank, 'float32'>(params);
    const outValue = gatherNdImplCPU(
        indicesData, paramsBuf, params.dtype, numSlices, sliceRank, sliceSize,
        strides, params.shape, paramsSize);

    return backend.makeTensorInfo(resultShape, params.dtype, outValue.values);
  }
  const program = new GatherNDProgram(sliceRank, [numSlices, sliceSize]);
  const uniformData =
      [{type: 'int32', data: [sliceRank]}, {type: 'int32', data: strides}];
  const res = backend.runWebGPUProgram(
      program, [flattenX, flattenIndices], flattenX.dtype, uniformData);

  const reshaped =
      reshape({inputs: {x: res}, backend, attrs: {shape: resultShape}});

  backend.disposeData(flattenIndices.dataId);
  backend.disposeData(flattenX.dataId);
  backend.disposeData(res.dataId);

  return reshaped;
}

export const gatherNdConfig: KernelConfig = {
  kernelName: GatherNd,
  backendName: 'webgpu',
  kernelFunc: gatherNd as {} as KernelFunc
};
