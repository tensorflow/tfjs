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

import {DataType, TypedArray, util} from '@tensorflow/tfjs-core';

export function sparseReshapeImpl(
    inputIndices: TypedArray, inputIndicesShape: number[], inputDType: DataType,
    inputShape: number[],
    targetShape: number[]): [TypedArray, number[], number[]] {
  const denseSize = util.sizeFromShape(inputShape);
  const nnz = inputIndicesShape[0];
  const outputRank = targetShape.length;

  // Compute the output shape. Determine product of specified dimensions, and
  // find the index of the unspecified one.
  const outputShape: number[] = [];
  let product = 1;
  let unknownIndex = -1;
  for (let d = 0; d < outputRank; ++d) {
    const size = targetShape[d];
    if (size === -1) {
      if (unknownIndex !== -1) {
        throw new Error(`only one output dimension may be -1, not both ${
            unknownIndex} and ${d}`);
      }
      unknownIndex = d;
      outputShape.push(1);
    } else {
      if (size < 0) {
        throw new Error(`size ${d} must be non-negative, not ${size}`);
      }
      product *= size;
      outputShape.push(size);
    }
  }
  if (unknownIndex !== -1) {
    if (product <= 0) {
      throw new Error(
          'reshape cannot infer the missing ' +
          'input size for an empty tensor unless all ' +
          'specified input sizes are non-zero');
    }
    const missing = Math.trunc(denseSize / product);
    if (product * missing !== denseSize) {
      throw new Error(`Input to reshape is a SparseTensor with ${denseSize}
          dense values, but the requested shape requires a multiple of ${
          product}. inputShape=${inputShape} outputShape= ${outputShape}`);
    }

    outputShape[unknownIndex] = missing;
  }
  const outputSize = util.sizeFromShape(outputShape);
  if (outputSize !== denseSize) {
    throw new Error(`Input to reshape is a tensor with ${
        denseSize} dense values, but the requested shape has ${
        outputSize}. inputShape=${inputShape} outputShape=${outputShape}`);
  }

  const inputRank = inputShape.length;
  const inputStrides: number[] = [];
  if (inputRank > 0) {
    inputStrides[inputRank - 1] = 1;
    for (let d = inputRank - 2; d >= 0; --d) {
      inputStrides[d] = inputStrides[d + 1] * inputShape[d + 1];
    }
  }

  const outputStrides: number[] = [];
  if (outputRank > 0) {
    outputStrides[outputRank - 1] = 1;
    for (let d = outputRank - 2; d >= 0; --d) {
      outputStrides[d] = outputStrides[d + 1] * outputShape[d + 1];
    }
  }

  const newIndices =
      util.getArrayFromDType(inputDType, nnz * outputRank) as TypedArray;
  for (let i = 0; i < nnz; ++i) {
    let id = 0;
    for (let j = 0; j < inputRank; ++j) {
      // inputIndices is a 2d tensor with shape of [nnz, inputRank]
      id += inputIndices[i * inputRank + j] * inputStrides[j];
    }
    for (let j = 0; j < outputRank; ++j) {
      // newIndices is a 2d tensor with shape of [nnz, outputRank]
      newIndices[i * outputRank + j] = Math.trunc(id / outputStrides[j]);
      id %= outputStrides[j];
    }
  }
  return [newIndices, [nnz, outputRank], outputShape];
}
