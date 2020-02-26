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

import {TensorInfo} from '../../../kernel_registry';

// TODO(yassogba) export from core
import {assertAndGetBroadcastShape, getBroadcastDims} from '../../../ops/broadcast_util';
import {DataType, NumericDataType, TypedArray} from '../../../types';
import {computeStrides, getTypedArrayFromDType, sizeFromShape} from '../../../util';
import {indexToLoc, locToIndex} from '../../../util';

import {MathBackendCPU} from '../backend_cpu';

export function broadcastedBinaryOp(
    a: TensorInfo, b: TensorInfo, dtype: DataType, backend: MathBackendCPU,
    op: (a: number, b: number) => number): TensorInfo {
  const newShape = assertAndGetBroadcastShape(a.shape, b.shape);

  const resultRank = newShape.length;
  const resultStrides = computeStrides(newShape);
  const resultSize = sizeFromShape(newShape);

  const result = getTypedArrayFromDType(dtype as NumericDataType, resultSize);

  const aVals = backend.data.get(a.dataId).values as TypedArray;
  const bVals = backend.data.get(b.dataId).values as TypedArray;

  const aBroadcastDims = getBroadcastDims(a.shape, newShape);
  const bBroadcastDims = getBroadcastDims(b.shape, newShape);

  if (aBroadcastDims.length + bBroadcastDims.length === 0) {
    for (let i = 0; i < result.length; ++i) {
      result[i] = op(aVals[i % aVals.length], bVals[i % bVals.length]);
    }
  } else {
    for (let i = 0; i < result.length; ++i) {
      const loc = indexToLoc(i, resultRank, resultStrides);

      const aLoc = loc.slice(-a.rank);
      aBroadcastDims.forEach(d => aLoc[d] = 0);
      const aIndex = locToIndex(aLoc, a.rank, a.strides);

      const bLoc = loc.slice(-b.rank);
      bBroadcastDims.forEach(d => bLoc[d] = 0);
      const bIndex = locToIndex(bLoc, b.rank, b.strides);

      result[i] = op(aVals[aIndex], bVals[bIndex]);
    }
  }

  const dataId = backend.write(result, newShape, dtype);
  return {
    dataId,
    shape: newShape,
    dtype,
    strides: resultStrides,
    rank: resultRank
  };
}
