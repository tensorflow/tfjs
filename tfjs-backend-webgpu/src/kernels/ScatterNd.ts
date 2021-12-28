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

import {backend_util, KernelConfig, KernelFunc, ScatterNd, ScatterNdAttrs, ScatterNdInputs, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {fill} from './Fill';
import {reshape} from './Reshape';
import {ScatterOptimizedProgram} from '../scatter_optimized_webgpu';

export function scatterNd(args: {
  inputs: ScatterNdInputs,
  backend: WebGPUBackend,
  attrs: ScatterNdAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {indices, updates} = inputs;
  const {shape} = attrs;

  const {sliceRank, numUpdates, sliceSize, strides, outputSize} =
      backend_util.calculateShapes(updates, indices, shape);

  const flattenShape = [outputSize / sliceSize, sliceSize];

  if (outputSize === 0) {
    return backend.makeTensorInfo(shape, indices.dtype);
  }

  const flattenIndices = reshape(
      {inputs: {x: indices}, backend, attrs: {shape: [numUpdates, sliceRank]}});
  const flattenX = reshape(
      {inputs: {x: updates}, backend, attrs: {shape: [numUpdates, sliceSize]}});

  const type = flattenX.dtype;
  const output =
      fill({backend, attrs: {shape: flattenShape, value: 0, dtype: type}});
  const size = util.sizeFromShape(flattenX.shape);
  const uniformData = [
    {type: 'int32', data: [sliceRank]}, {type: 'int32', data: strides},
    {type: 'int32', data: [size]}
  ];
  const program = new ScatterOptimizedProgram(
      flattenX.shape, sliceRank, flattenIndices.shape.length,
      flattenX.shape.length, strides, flattenShape, type);
  const res = backend.runWebGPUProgram(
      program, [flattenX, flattenIndices], type, uniformData, output);

  const reshaped = reshape({inputs: {x: res}, backend, attrs: {shape}});

  backend.disposeData(flattenIndices.dataId);
  backend.disposeData(flattenX.dataId);
  backend.disposeData(res.dataId);

  return reshaped;
}

export const scatterNdConfig: KernelConfig = {
  kernelName: ScatterNd,
  backendName: 'webgpu',
  kernelFunc: scatterNd as {} as KernelFunc
};
