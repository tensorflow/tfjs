/**
 * @license
 * Copyright 2022 Google LLC.
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

const INT32_MAX = 2147483647;

export function raggedRangeImpl(
    starts: TypedArray, startsShape: number[], startsDType: DataType,
    limits: TypedArray, limitsShape: number[], deltas: TypedArray,
    deltasShape: number[]): [TypedArray, TypedArray] {
  // Check input tensor shapes.
  if (startsShape.length > 1) {
    throw new Error('starts must be a scalar or vector');
  }
  if (limitsShape.length > 1) {
    throw new Error('limits must be a scalar or vector');
  }
  if (deltasShape.length > 1) {
    throw new Error('deltas must be a scalar or vector');
  }

  // Determine which tensors we need to broadcast.
  const broadcastStarts = startsShape.length === 0;
  const broadcastLimits = limitsShape.length === 0;
  const broadcastDeltas = deltasShape.length === 0;

  // nRows (number of output rows) is the size of the non-broadcast inputs,
  // or 1 if all inputs are scalars.
  const inSizes: number[] = [];
  if (!broadcastStarts) {
    inSizes.push(startsShape[0]);
  }
  if (!broadcastLimits) {
    inSizes.push(limitsShape[0]);
  }
  if (!broadcastDeltas) {
    inSizes.push(deltasShape[0]);
  }

  for (let i = 1; i < inSizes.length; ++i) {
    if (inSizes[i] !== inSizes[i - 1]) {
      throw new Error('starts, limits, and deltas must have the same shape');
    }
  }
  const nRows = inSizes.length === 0 ? 1 : inSizes[0];

  // Construct the rtNestedSplits tensor.
  const rtNestedSplits =
      util.getArrayFromDType('int32', nRows + 1) as TypedArray;
  rtNestedSplits[0] = 0;
  for (let row = 0; row < nRows; ++row) {
    const start = broadcastStarts ? starts[0] : starts[row];
    const limit = broadcastLimits ? limits[0] : limits[row];
    const delta = broadcastDeltas ? deltas[0] : deltas[row];
    if (delta === 0) {
      throw new Error('Requires delta != 0');
    }
    let size: number;  // The number of elements in the specified range.
    if (((delta > 0) && (limit < start)) || ((delta < 0) && (limit > start))) {
      size = 0;
    } else {
      size = Math.ceil(Math.abs((limit - start) / delta));

      if (size > INT32_MAX) {
        throw new Error(`Requires ((limit - start) / delta) <= ${INT32_MAX}`);
      }
    }
    rtNestedSplits[row + 1] = rtNestedSplits[row] + size;
  }

  const nVals = rtNestedSplits[nRows];

  // Construct the rtDenseValues tensor.
  const rtDenseValues =
      util.getArrayFromDType(startsDType, nVals) as TypedArray;

  let valueIndex = 0;
  for (let row = 0; row < nRows; ++row) {
    const rowSize = rtNestedSplits[row + 1] - rtNestedSplits[row];
    let value = broadcastStarts ? starts[0] : starts[row];
    const delta = broadcastDeltas ? deltas[0] : deltas[row];
    for (let i = 0; i < rowSize; ++i) {
      rtDenseValues[valueIndex++] = value;
      value += delta;
    }
  }

  return [rtNestedSplits, rtDenseValues];
}
