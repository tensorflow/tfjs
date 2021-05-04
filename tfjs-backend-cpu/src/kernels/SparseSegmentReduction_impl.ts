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

export function sparseSegmentReductionImpl(
    input: TypedArray, inputShape: number[], inputDType: DataType,
    indices: TypedArray, segmentIds: TypedArray, isMean = false,
    defaultValue = 0): [TypedArray, number[]] {
  const numIndices = indices.length;
  if (numIndices !== segmentIds.length) {
    throw new Error(`segmentIds and indices should have same size.`);
  }

  // Flatten the array to two dimensions
  const inputFlat: number[] = [inputShape[0], input.length / inputShape[0]];
  const numCol = inputFlat[1];
  // Note that the current implementation assumes that segmentIds values are
  // sorted.
  const lastSegmentIdPlusOne =
      numIndices > 0 ? segmentIds[numIndices - 1] + 1 : 0;
  const outputRows = lastSegmentIdPlusOne;

  if (outputRows < 0) {
    throw new Error(`segment ids must be >= 0`);
  }

  const outputShape = inputShape.slice();
  outputShape[0] = outputRows;

  const outputLength =
      outputShape.reduce((product, value) => product * value, 1);
  // Output array is initialized with the value 0 by default.
  const output = util.getArrayFromDType(inputDType, outputLength) as TypedArray;

  // Note that we do not initialize the output buffer with a default value, so
  // we need to explicitly set missing indices to the default value.
  if (numIndices === 0) {
    if (outputRows > 0) {
      output.fill(defaultValue);
    }
    return [output, outputShape];
  }

  if (outputRows <= 0) {
    throw new Error(`segment ids must be >= 0`);
  }

  let start = 0, end = 1;
  // Index from which the output is not initialized.
  let uninitializedIndex = 0;
  let outIndex = segmentIds[start];

  while (true) {
    // We initialize nextIndex to 0 to avoid may be uninitialized warning
    let nextIndex = 0;
    if (end < numIndices) {
      nextIndex = segmentIds[end];
      if (outIndex === nextIndex) {
        ++end;
        continue;
      }
      // We have a new segment here.  Verify that the segment ids are growing.
      if (outIndex >= nextIndex) {
        throw new Error(`segment ids are not increasing`);
      }
    }

    if (outIndex < 0 || outIndex >= outputRows) {
      throw new Error(`Segment id ${outIndex} out of range [0, ${
          outputRows}), possibly because segmentIds input is not sorted.`);
    }

    // If there is a gap between two indices, we need to set that gap to the
    // default value.
    if (outIndex > uninitializedIndex) {
      output.fill(defaultValue, uninitializedIndex * numCol, outIndex * numCol);
    }

    for (let i = start; i < end; ++i) {
      const index = indices[i];
      if (index < 0 || index >= inputFlat[0]) {
        throw new Error(`Bad: indices[${i}] == ${indices[i]} out of range [0, ${
            inputFlat[0]})`);
      }
      for (let j = 0; j < numCol; j++) {
        output[outIndex * numCol + j] += input[index * numCol + j];
      }
    }

    if (isMean) {
      for (let j = 0; j < numCol; j++) {
        output[outIndex * numCol + j] /= end - start;
      }
    }

    start = end;
    ++end;
    uninitializedIndex = outIndex + 1;
    outIndex = nextIndex;
    if (end > numIndices) {
      break;
    }
  }

  // Fill the gap at the end with the default value.
  if (uninitializedIndex < outputRows) {
    output.fill(defaultValue, uninitializedIndex * numCol, outputRows * numCol);
  }

  return [output, outputShape];
}
