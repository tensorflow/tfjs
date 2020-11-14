/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import {nearestDivisor} from '../util';

import {PARALLELIZE_THRESHOLD} from './reduce_util';

export interface SegOpInfo {
  windowSize: number;
  batchSize: number;
  inSize: number;
  numSegments: number;
}

export function segOpComputeOptimalWindowSize(
    inSize: number, numSegments: number): number {
  let done = false;
  let res;

  if (inSize <= PARALLELIZE_THRESHOLD) {
    res = inSize;
    done = true;
  } else {
    res = nearestDivisor(inSize, Math.floor(Math.sqrt(inSize)));
  }

  while (!done) {
    if (res > numSegments || res === inSize) {
      done = true;
    } else {
      res = nearestDivisor(inSize, res + 1);
    }
  }
  return res;
}

export function computeOutShape(
    aShape: number[], axis: number, numSegments: number): number[] {
  const outShape = [];
  const rank = aShape.length;
  for (let dim = 0; dim < rank; dim++) {
    if (dim !== axis) {
      outShape.push(aShape[dim]);
    } else {
      outShape.push(numSegments);
    }
  }
  return outShape;
}

export interface GatherOpShapeInfo {
  batchSize: number;
  sliceSize: number;
  outerSize: number;
  dimSize: number;
  outputShape: number[];
}

export function collectGatherOpShapeInfo(
    x: Tensor, indices: Tensor, axis: number,
    batchDims: number): GatherOpShapeInfo {
  if (batchDims != 0) {
    if (batchDims < -indices.rank || batchDims > indices.rank) {
      throw new Error(`Expect batchDims in the range of [-${indices.rank}, ${
          indices.rank}], but got ${batchDims}`);
    }
  }

  if (batchDims < 0) {
    batchDims += indices.rank;
  }

  if (batchDims > x.rank) {
    throw new Error(`batchDims (${batchDims}) must be less than rank(x) (
    ${x.rank}).`);
  }


  if (axis < batchDims) {
    throw new Error(`batchDims (${
        batchDims}) must be less than or equal to axis (${axis}).`);
  }

  for (let i = 0; i < batchDims; ++i) {
    if (x.shape[i] !== indices.shape[i]) {
      throw new Error(
          `x.shape[${i}]: ${x.shape[i]} should be equal to indices.shape[${
              i}]: ${indices.shape[i]}.`);
    }
  }
  const dimSize = x.shape[axis];

  const outputShape: number[] = [];
  let batchSize = 1;
  let outerSize = 1;
  let sliceSize = 1;

  for (let i = 0; i < batchDims; ++i) {
    outputShape.push(x.shape[i]);
    batchSize *= x.shape[i];
  }

  for (let i = batchDims; i < axis; i++) {
    outputShape.push(x.shape[i]);
    outerSize *= x.shape[i];
  }

  for (let i = batchDims; i < indices.rank; i++) {
    outputShape.push(indices.shape[i]);
  }

  for (let i = axis + 1; i < x.rank; i++) {
    outputShape.push(x.shape[i]);
    sliceSize *= x.shape[i];
  }

  return {batchSize, sliceSize, outerSize, dimSize, outputShape};
}
