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

import {backend_util, GatherNd, GatherNdInputs, KernelConfig, KernelFunc, TensorInfo, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {GatherNDProgram} from '../gather_nd_gpu';
import {reshape} from './Reshape';

export function gatherNd(
    args: {inputs: GatherNdInputs, backend: MathBackendWebGL}): TensorInfo {
  const {inputs, backend} = args;
  const {params, indices} = inputs;

  const indicesShape = indices.shape;
  const sliceRank = indicesShape[indicesShape.length - 1];

  const [resultShape, numSlices, sliceSize, strides] =
      backend_util.prepareAndValidate(params, indices);

  const flattenIndices = reshape(
      {inputs: {x: indices}, backend, attrs: {shape: [numSlices, sliceRank]}});
  const flattenX = reshape({
    inputs: {x: params},
    backend,
    attrs: {shape: [(util.sizeFromShape(params.shape) / sliceSize), sliceSize]}
  });

  const program =
      new GatherNDProgram(sliceRank, strides, [numSlices, sliceSize]);
  const res = backend.runWebGLProgram(
      program, [flattenX, flattenIndices], flattenX.dtype);

  const reshaped =
      reshape({inputs: {x: res}, backend, attrs: {shape: resultShape}});

  backend.disposeIntermediateTensorInfo(flattenIndices);
  backend.disposeIntermediateTensorInfo(flattenX);
  backend.disposeIntermediateTensorInfo(res);

  return reshaped;
}

export const gatherNdConfig: KernelConfig = {
  kernelName: GatherNd,
  backendName: 'webgl',
  kernelFunc: gatherNd as {} as KernelFunc
};
