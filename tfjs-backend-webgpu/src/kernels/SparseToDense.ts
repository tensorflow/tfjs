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

import {backend_util, KernelConfig, KernelFunc, SparseToDense, SparseToDenseAttrs, SparseToDenseInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {reshape} from './Reshape';
import {ScatterProgram} from './scatter_webgpu';

export function sparseToDense(args: {
  inputs: SparseToDenseInputs,
  backend: WebGPUBackend,
  attrs: SparseToDenseAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {sparseIndices, sparseValues, defaultValue} = inputs;
  const {outputShape} = attrs;

  const {sliceRank, numUpdates, strides, outputSize} =
      backend_util.calculateShapes(sparseValues, sparseIndices, outputShape);

  const sumDupeIndices = false;
  const program = new ScatterProgram(
      numUpdates, sliceRank, sparseIndices.shape.length,
      sparseValues.shape.length, strides, [outputSize, 1], sumDupeIndices);

  const res = backend.runWebGPUProgram(
      program, [sparseValues, sparseIndices, defaultValue], sparseValues.dtype);

  const reshaped =
      reshape({inputs: {x: res}, backend, attrs: {shape: outputShape}});

  backend.disposeData(res.dataId);
  return reshaped;
}

export const sparseToDenseConfig: KernelConfig = {
  kernelName: SparseToDense,
  backendName: 'webgpu',
  kernelFunc: sparseToDense as {} as KernelFunc
};
