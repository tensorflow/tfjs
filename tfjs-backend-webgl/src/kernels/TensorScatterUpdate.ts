/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {backend_util, KernelConfig, KernelFunc, TensorInfo, TensorScatterUpdate, TensorScatterUpdateAttrs, TensorScatterUpdateInputs} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {ScatterProgram} from '../scatter_gpu';

import {reshape} from './Reshape';

export function tensorScatterUpdate(args: {
  inputs: TensorScatterUpdateInputs,
  backend: MathBackendWebGL,
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

  const flattenIndices = reshape(
      {inputs: {x: indices}, backend, attrs: {shape: [numUpdates, sliceRank]}});
  const flattenX = reshape(
      {inputs: {x: updates}, backend, attrs: {shape: [numUpdates, sliceSize]}});
  const flattenTensor =
      reshape({inputs: {x: tensor}, backend, attrs: {shape: flattenShape}});
  const program = new ScatterProgram(
      numUpdates, sliceRank, flattenIndices.shape.length, flattenX.shape.length,
      strides, flattenShape, false, true);
  const res = backend.runWebGLProgram(
      program, [flattenX, flattenIndices, flattenTensor], flattenTensor.dtype);

  const reshaped =
      reshape({inputs: {x: res}, backend, attrs: {shape: tensor.shape}});

  backend.disposeIntermediateTensorInfo(flattenIndices);
  backend.disposeIntermediateTensorInfo(flattenX);
  backend.disposeIntermediateTensorInfo(flattenTensor);
  backend.disposeIntermediateTensorInfo(res);

  return reshaped;
}

export const tensorScatterUpdateConfig: KernelConfig = {
  kernelName: TensorScatterUpdate,
  backendName: 'webgl',
  kernelFunc: tensorScatterUpdate as unknown as KernelFunc
};
