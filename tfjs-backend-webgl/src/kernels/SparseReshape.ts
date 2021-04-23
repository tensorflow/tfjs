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

import {KernelConfig, SparseReshape, SparseReshapeInputs, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {sparseReshapeImplCPU} from '../kernel_utils/shared';

export function sparseReshape(
    args: {inputs: SparseReshapeInputs, backend: MathBackendWebGL}):
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

  const $inputShape =
      Array.from(backend.readSync(inputShape.dataId) as TypedArray);
  const $inputIndices = backend.readSync(inputIndices.dataId) as TypedArray;
  const targetShape =
      Array.from(backend.readSync(newShape.dataId) as TypedArray);

  const [newIndices, indicesShape, outputShape] = sparseReshapeImplCPU(
      $inputIndices, inputIndices.shape, inputIndices.dtype, $inputShape,
      targetShape);
  return [
    backend.makeTensorInfo(indicesShape, inputIndices.dtype, newIndices),
    backend.makeTensorInfo(
        [outputShape.length], newShape.dtype, new Int32Array(outputShape)),
  ];
}

export const sparseReshapeConfig: KernelConfig = {
  kernelName: SparseReshape,
  backendName: 'webgl',
  kernelFunc: sparseReshape,
};
