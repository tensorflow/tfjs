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

import {buffer, Rank, TensorBuffer} from '@tensorflow/tfjs-core';

export function stridedSliceImpl<R extends Rank>(
    outShape: number[], xBuf: TensorBuffer<R>, strides: number[],
    begin: number[]): TensorBuffer<R> {
  const outBuf = buffer(outShape, xBuf.dtype);

  for (let i = 0; i < outBuf.size; i++) {
    const loc = outBuf.indexToLoc(i);

    const newLoc: number[] = new Array(loc.length);
    for (let j = 0; j < newLoc.length; j++) {
      newLoc[j] = loc[j] * strides[j] + begin[j];
    }
    outBuf.set(xBuf.get(...newLoc), ...loc);
  }

  return outBuf as TensorBuffer<R>;
}
