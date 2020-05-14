/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {DataType, NumericDataType, TypedArray} from '@tensorflow/tfjs-core';
import {util} from '@tensorflow/tfjs-core';

export function transposeImpl(
    xVals: TypedArray, xShape: number[], dtype: DataType, perm: number[],
    newShape: number[]): TypedArray {
  const xRank = xShape.length;
  const xSize = util.sizeFromShape(xShape);
  const xStrides = util.computeStrides(xShape);
  const newStrides = util.computeStrides(newShape);

  const result = util.getTypedArrayFromDType(
      dtype as NumericDataType, util.sizeFromShape(newShape));

  for (let i = 0; i < xSize; ++i) {
    const loc = util.indexToLoc(i, xRank, xStrides);

    // Permute location.
    const newLoc: number[] = new Array(loc.length);
    for (let i = 0; i < newLoc.length; i++) {
      newLoc[i] = loc[perm[i]];
    }

    const newIndex = util.locToIndex(newLoc, xRank, newStrides);
    result[newIndex] = xVals[i];
  }
  return result;
}
