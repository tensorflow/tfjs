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

import {backend_util, KernelConfig, SparseReshape, SparseReshapeInputs, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {SparseReshapeOutputIndicesProgram, SparseReshapeOutputShapeProgram} from '../sparse_reshape_webgpu';

export function sparseReshape(
    args: {inputs: SparseReshapeInputs, backend: WebGPUBackend}):
    [TensorInfo, TensorInfo] {
  const {inputs, backend} = args;
  const {inputIndices, inputShape, newShape} = inputs;

  const shapeOnCPU = backend.shouldExecuteOnCPU([inputShape, newShape]);
  let outputShape: TensorInfo;
  if (shapeOnCPU) {
    const $inputShape = Array.from(
        backend.tensorMap.get(inputShape.dataId).values as TypedArray);
    const $newShape =
        Array.from(backend.tensorMap.get(newShape.dataId).values as TypedArray);

    const denseSize = util.sizeFromShape($inputShape);
    const outputRank = $newShape.length;

    // Compute the output shape. Determine product of specified dimensions, and
    // find the index of the unspecified one.
    const shape: number[] = [];
    let product = 1;
    let unknownIndex = -1;
    for (let d = 0; d < outputRank; ++d) {
      const size = $newShape[d];
      if (size === -1) {
        if (unknownIndex !== -1) {
          throw new Error(
              backend_util
                  .getSparseReshapeMultipleNegativeOneOutputDimErrorMessage(
                      unknownIndex, d));
        }
        unknownIndex = d;
        shape.push(1);
      } else {
        if (size < 0) {
          throw new Error(
              backend_util.getSparseReshapeNegativeOutputDimErrorMessage(
                  d, size));
        }
        product *= size;
        shape.push(size);
      }
    }
    if (unknownIndex !== -1) {
      if (product <= 0) {
        throw new Error(
            backend_util
                .getSparseReshapeEmptyTensorZeroOutputDimErrorMessage());
      }
      const missing = Math.trunc(denseSize / product);
      if (product * missing !== denseSize) {
        throw new Error(
            backend_util.getSparseReshapeInputOutputMultipleErrorMessage(
                $inputShape, shape));
      }

      shape[unknownIndex] = missing;
    }
    const outputSize = util.sizeFromShape(shape);
    if (outputSize !== denseSize) {
      throw new Error(
          backend_util.getSparseReshapeInputOutputMismatchErrorMessage(
              $inputShape, shape));
    }
    outputShape = backend.makeTensorInfo(
        [shape.length], newShape.dtype, new Int32Array(shape));
  } else {
    const shapeProgram = new SparseReshapeOutputShapeProgram(newShape.shape);
    outputShape = backend.runWebGPUProgram(
        shapeProgram, [inputShape, newShape], newShape.dtype);
  }

  const indicesProgram = new SparseReshapeOutputIndicesProgram(
      [inputIndices.shape[0], newShape.shape[0]]);

  const outputIndices = backend.runWebGPUProgram(
      indicesProgram, [inputIndices, inputShape, outputShape],
      inputIndices.dtype);

  return [outputIndices, outputShape];
}

export const sparseReshapeConfig: KernelConfig = {
  kernelName: SparseReshape,
  backendName: 'webgpu',
  kernelFunc: sparseReshape,
};
