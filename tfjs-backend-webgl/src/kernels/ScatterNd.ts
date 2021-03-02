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

import {backend_util, KernelConfig, KernelFunc, ScatterNd, ScatterNdAttrs, ScatterNdInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {ScatterProgram} from '../scatter_gpu';
import {reshape} from './Reshape';

export function scatterNd(args: {
  inputs: ScatterNdInputs,
  backend: MathBackendWebGL,
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

  const defaultValue = backend.makeTensorInfo(
      [], 'float32', new Float32Array([0]));  // scalar(0)
  const program = new ScatterProgram(
      numUpdates, sliceRank, flattenIndices.shape.length, flattenX.shape.length,
      strides, flattenShape);
  const res = backend.runWebGLProgram(
      program, [flattenX, flattenIndices, defaultValue], flattenX.dtype);

  const reshaped = reshape({inputs: {x: res}, backend, attrs: {shape}});

  backend.disposeIntermediateTensorInfo(flattenIndices);
  backend.disposeIntermediateTensorInfo(flattenX);
  backend.disposeIntermediateTensorInfo(res);
  backend.disposeIntermediateTensorInfo(defaultValue);

  return reshaped;
}

export const scatterNdConfig: KernelConfig = {
  kernelName: ScatterNd,
  backendName: 'webgl',
  kernelFunc: scatterNd as {} as KernelFunc
};
