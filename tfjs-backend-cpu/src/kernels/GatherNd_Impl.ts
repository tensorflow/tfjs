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

import {buffer, DataType, Rank, TensorBuffer, TypedArray} from '@tensorflow/tfjs-core';

export function gatherNdImpl<R extends Rank>(
    indicesData: TypedArray, paramsBuf: TensorBuffer<R>, dtype: DataType,
    numSlices: number, sliceRank: number, sliceSize: number, strides: number[],
    paramsShape: number[], paramsSize: number): TensorBuffer<R> {
  const outBuf = buffer([numSlices, sliceSize], dtype);

  for (let i = 0; i < numSlices; i++) {
    const index = [];
    let flattenIndex = 0;
    for (let j = 0; j < sliceRank; j++) {
      const dim = indicesData[i * sliceRank + j];
      flattenIndex += dim * strides[j];
      index.push(dim);
    }
    if (flattenIndex < 0 || flattenIndex >= paramsSize / sliceSize) {
      throw new Error(
          `Invalid indices: ${index} does not index into ${paramsShape}`);
    }

    for (let k = 0; k < sliceSize; k++) {
      outBuf.values[i * sliceSize + k] =
          paramsBuf.get(...paramsBuf.indexToLoc(flattenIndex * sliceSize + k));
    }
  }

  return outBuf as TensorBuffer<R>;
}
