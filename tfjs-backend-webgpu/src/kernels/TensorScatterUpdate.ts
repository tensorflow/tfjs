/**
 * @license
 * Copyright 2023 Google LLC.
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

import {backend_util, KernelConfig, KernelFunc, TensorInfo, TensorScatterUpdate, TensorScatterUpdateAttrs, TensorScatterUpdateInputs, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {ScatterProgram} from '../scatter_webgpu';

import {reshape} from './Reshape';
import {tile} from './Tile';

export function tensorScatterUpdate(args: {
  inputs: TensorScatterUpdateInputs,
  backend: WebGPUBackend,
  attrs: TensorScatterUpdateAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {tensor, indices, updates} = inputs;
  const {} = attrs;

  const {sliceRank, numUpdates, sliceSize, strides, outputSize} =
      backend_util.calculateShapes(updates, indices, tensor.shape);

  const flattenShape = [outputSize / sliceSize, sliceSize];

  if (outputSize === 0) {
    return backend.makeTensorInfo(tensor.shape, indices.dtype);
  }

  const toDispose = [];

  const flattenIndices = reshape(
      {inputs: {x: indices}, backend, attrs: {shape: [numUpdates, sliceRank]}});
  toDispose.push(flattenIndices);
  const flattenX = reshape(
      {inputs: {x: updates}, backend, attrs: {shape: [numUpdates, sliceSize]}});
  toDispose.push(flattenX);
  const flattenTensor =
      reshape({inputs: {x: tensor}, backend, attrs: {shape: flattenShape}});
  toDispose.push(flattenTensor);
  const output = tile({
    inputs: {x: flattenTensor},
    backend,
    attrs: {reps: Array(flattenShape.length).fill(1)}
  });
  const program = new ScatterProgram(
      [numUpdates, sliceSize], sliceRank, flattenIndices.shape.length,
      flattenX.shape.length, strides, flattenShape, tensor.dtype, false);
  const size = util.sizeFromShape([numUpdates, sliceSize]);
  const uniformData = [
    {type: 'int32', data: [sliceRank]},
    {type: 'int32', data: strides},
    {type: 'int32', data: [size]},
  ];
  const res = backend.runWebGPUProgram(
      program, [flattenX, flattenIndices], flattenTensor.dtype, uniformData,
      output);
  toDispose.push(res);

  const reshaped =
      reshape({inputs: {x: res}, backend, attrs: {shape: tensor.shape}});

  toDispose.forEach(t => backend.disposeData(t.dataId));

  return reshaped;
}

export const tensorScatterUpdateConfig: KernelConfig = {
  kernelName: TensorScatterUpdate,
  backendName: 'webgpu',
  kernelFunc: tensorScatterUpdate as unknown as KernelFunc
};
