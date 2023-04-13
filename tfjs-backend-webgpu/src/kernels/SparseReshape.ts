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

import {KernelConfig, SparseReshape, SparseReshapeInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {SparseReshapeOutputIndicesProgram, SparseReshapeOutputShapeProgram} from '../sparse_reshape_webgpu';

export function sparseReshape(
    args: {inputs: SparseReshapeInputs, backend: WebGPUBackend}):
    [TensorInfo, TensorInfo] {
  const {inputs, backend} = args;
  const {inputIndices, inputShape, newShape} = inputs;
  if (inputIndices.shape.length !== 2) {
    throw new Error(`Input indices should be a matrix but received shape ${
        inputIndices.shape}`);
  }
  if (inputShape.shape.length !== 1) {
    throw new Error(`Input shape should be a vector but received shape ${
        inputShape.shape}`);
  }

  if (newShape.shape.length !== 1) {
    throw new Error(
        `Target shape should be a vector but received shape ${newShape.shape}`);
  }

  let program = new SparseReshapeOutputShapeProgram(newShape.shape);
  const outputShape =
      backend.runWebGPUProgram(program, [inputShape, newShape], newShape.dtype);

  program = new SparseReshapeOutputIndicesProgram(
      [inputIndices.shape[0], newShape.shape[0]]);
  const outputIndices = backend.runWebGPUProgram(
      program, [inputIndices, inputShape, outputShape], inputIndices.dtype);
  return [outputIndices, outputShape];
}

export const sparseReshapeConfig: KernelConfig = {
  kernelName: SparseReshape,
  backendName: 'webgpu',
  kernelFunc: sparseReshape,
};
