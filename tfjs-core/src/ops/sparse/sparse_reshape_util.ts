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
import {sizeFromShape} from '../../util';

/**
 * Generates sparse reshape multiple negative 1 output dimension error message.
 *
 * @param dim1 The first dimension with a negative 1 value.
 * @param dim2 The second dimension with a negative 1 value.
 */
export function getSparseReshapeMultipleNegativeOneOutputDimErrorMessage(
    dim1: number, dim2: number) {
  return `only one output dimension may be -1, not both ${dim1} and ${dim2}`;
}

/**
 * Generates sparse reshape negative output dimension error message.
 *
 * @param dim The dimension with a negative value.
 * @param value The negative value.
 */
export function getSparseReshapeNegativeOutputDimErrorMessage(
    dim: number, value: number) {
  return `size ${dim} must be non-negative, not ${value}`;
}

/**
 * Generates sparse reshape empty tensor zero output dimension error message.
 *
 */
export function getSparseReshapeEmptyTensorZeroOutputDimErrorMessage() {
  return 'reshape cannot infer the missing input size for an empty tensor ' +
      'unless all specified input sizes are non-zero';
}

/**
 * Generates sparse reshape input output multiple mismatch error message.
 *
 * @param inputShape the input shape.
 * @param outputShape the requested output shape.
 */
export function getSparseReshapeInputOutputMultipleErrorMessage(
    inputShape: number[], outputShape: number[]) {
  const inputSize = sizeFromShape(inputShape);
  const outputSize = sizeFromShape(outputShape);
  return `Input to reshape is a SparseTensor with ${inputSize}
  dense values, but the requested shape requires a multiple of ${
      outputSize}. inputShape=${inputShape} outputShape= ${outputShape}`;
}

/**
 * Generates sparse reshape input output inequality error message.
 *
 * @param inputShape the input shape.
 * @param outputShape the requested output shape.
 */
export function getSparseReshapeInputOutputMismatchErrorMessage(
    inputShape: number[], outputShape: number[]) {
  const inputSize = sizeFromShape(inputShape);
  const outputSize = sizeFromShape(outputShape);
  return `Input to reshape is a tensor with ${
      inputSize} dense values, but the requested shape has ${
      outputSize}. inputShape=${inputShape} outputShape=${outputShape}`;
}
