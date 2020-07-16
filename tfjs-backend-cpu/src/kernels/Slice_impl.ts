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

import {DataType, DataTypeMap, Rank, slice_util, TensorBuffer, TypedArray, util} from '@tensorflow/tfjs-core';

export function sliceContinuousImpl(
    xVals: TypedArray|Uint8Array[], begin: number[], size: number[],
    dtype: string, strides: number[]): DataTypeMap[DataType] {
  const flatOffset = slice_util.computeFlatOffset(begin, strides);
  const length = util.sizeFromShape(size);

  if (dtype === 'string') {
    const decodedVals =
        (xVals as Uint8Array[]).map(val => util.decodeString(val));
    return decodedVals.slice(flatOffset, flatOffset + length);
  } else {
    return (xVals as TypedArray).subarray(flatOffset, flatOffset + length);
  }
}

export function sliceImpl<R extends Rank>(
    xBuf: TensorBuffer<R, DataType>, begin: number[],
    buffer: TensorBuffer<R, DataType>) {
  for (let i = 0; i < buffer.size; ++i) {
    const loc = buffer.indexToLoc(i);
    const xLoc = loc.map((idx, j) => idx + begin[j]);
    buffer.values[i] = xBuf.get(...xLoc) as DataType;
  }
}
