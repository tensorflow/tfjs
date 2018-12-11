/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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
import {Tensor} from '../tensor';
import {computeStrides, sizeFromShape} from '../util';

/**
 * Check whether updates.shape = indices.shape[:batchDim] +
 * shape[sliceDim:]
 *
 * @param x The input tensor.
 */
export function validateUpdateShape(
    shape: number[], indices: Tensor, updates: Tensor) {
  const sliceDim = (indices.rank > 1) ? indices.shape[indices.rank - 1] : 1;
  const batchDim = (indices.rank > 1) ? indices.rank - 1 : 1;

  const shapeError = 'Must have updates.shape = indices.shape[:batchDim] + ' +
      `shape[sliceDim:], got updates.shape: ${updates.shape}` +
      `, indices.shape: ${indices.shape}, shape: ${shape}` +
      `, sliceDim: ${sliceDim}, and batchDim: ${batchDim}.`;

  if (updates.rank < batchDim) {
    throw new Error(shapeError + ` update.rank < ${batchDim}. `);
  }
  if (shape.length < sliceDim + (updates.rank - batchDim)) {
    throw new Error(
        shapeError +
        ` Output shape length < ${sliceDim + (updates.rank - batchDim)}`);
  }
  if (updates.rank !== batchDim + shape.length - sliceDim) {
    throw new Error(
        shapeError + ` update.rank != ${batchDim + shape.length - sliceDim}`);
  }
  for (let d = 0; d < batchDim; ++d) {
    if (updates.shape[d] !== indices.shape[d]) {
      throw new Error(
          shapeError +
          ` updates.shape[${d}] (${updates.shape[d]}) != indices.shape[${d}] (${
              indices.shape[d]}).`);
    }
  }
  for (let d = 0; d < updates.rank - batchDim; ++d) {
    if (updates.shape[d + batchDim] !== shape[d + sliceDim]) {
      throw new Error(
          shapeError +
          ` updates.shape[${d + batchDim}] (${
              updates.shape[d + batchDim]}) != shape[${d + batchDim}] (${
              shape[d + batchDim]})`);
    }
  }
}

export interface ScatterShapeInfo {
  sliceRank: number;
  numUpdates: number;
  sliceSize: number;
  strides: number[];
  outputSize: number;
}
/**
 * Validate scatter nd inputs.
 *
 * @param update The tensor contains the update values.
 * @param indices The tensor contains the indices for the update values.
 * @param shape The shape of the output tensor.
 */
export function validateInput(
    updates: Tensor, indices: Tensor, shape: number[]) {
  if (indices.rank < 1) {
    throw new Error(
        'tf.scatterND() expects the indices to be rank 1 or higher,' +
        ` but the rank was ${indices.rank}.`);
  }
  if (updates.rank < 1) {
    throw new Error(
        'tf.scatterND() expects the updates to be rank 1 or higher,' +
        ` but the rank was ${updates.rank}.`);
  }
  if (indices.dtype !== 'int32') {
    throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${
        indices.dtype}`);
  }
  if (shape.length < 1) {
    throw new Error(
        `Output rank must be greater or equal to 1, but got shape: ${shape}`);
  }

  if (shape.length === 0) {
    if (indices.size === 0) {
      throw new Error(`Indices specified for empty output. indices shape: ${
          indices.shape}`);
    }
    if (updates.size === 0) {
      throw new Error(`Updates specified for empty output. updates shape: ${
          updates.shape}`);
    }
  }

  validateUpdateShape(shape, indices, updates);
}

/**
 * Calculate the shape information for the output.
 *
 * @param update The tensor contains the update values.
 * @param indices The tensor contains the indices for the update values.
 * @param shape The shape of the output tensor.
 *
 * @returns ScatterShapeInfo
 */
export function calculateShapes(
    updates: Tensor, indices: Tensor, shape: number[]): ScatterShapeInfo {
  // Calculate the number of dimensions in indices
  const sliceRank = (indices.rank > 1) ? indices.shape[indices.rank - 1] : 1;

  // Calculate the number of elements that make up each slice of our updated
  // tensor. This allows us to work with flattened tensors and copy over whole
  // slices at a time.
  const totalNd = shape.length;

  let sliceSize = 1;
  for (let i = sliceRank; i < totalNd; ++i) {
    sliceSize *= shape[i];
  }

  const safeSliceDim = (sliceRank < 1) ? 1 : sliceRank;
  const numUpdates = indices.size / safeSliceDim;

  const strides = [...computeStrides(shape.slice(0, sliceRank)), 1];
  const outputSize = sizeFromShape(shape);
  return {sliceRank, numUpdates, sliceSize, strides, outputSize};
}
