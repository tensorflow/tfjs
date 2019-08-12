/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

/**
 * An implementation of the tile kernel shared between webgl and cpu for string
 * tensors only.
 */

import {buffer} from '../ops/array_ops';
import {Tensor, TensorBuffer} from '../tensor';
import {DataType, Rank} from '../types';

export function tile<R extends Rank>(
    xBuf: TensorBuffer<R, DataType>, reps: number[]): Tensor<R> {
  const newShape: number[] = new Array(xBuf.rank);
  for (let i = 0; i < newShape.length; i++) {
    newShape[i] = xBuf.shape[i] * reps[i];
  }
  const result = buffer(newShape, xBuf.dtype);
  for (let i = 0; i < result.values.length; ++i) {
    const newLoc = result.indexToLoc(i);

    const originalLoc: number[] = new Array(xBuf.rank);
    for (let i = 0; i < originalLoc.length; i++) {
      originalLoc[i] = newLoc[i] % xBuf.shape[i];
    }

    const originalIndex = xBuf.locToIndex(originalLoc);

    result.values[i] = xBuf.values[originalIndex];
  }
  return result.toTensor() as Tensor<R>;
}
