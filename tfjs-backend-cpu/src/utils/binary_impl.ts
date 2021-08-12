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

import {backend_util, DataType, DataValues, NumericDataType, TypedArray, util} from '@tensorflow/tfjs-core';

import {SimpleBinaryKernelImpl, SimpleBinaryOperation} from './binary_types';

/**
 * Template that creates implementation for binary ops. Supports broadcast.
 */
export function createSimpleBinaryKernelImpl(op: SimpleBinaryOperation):
    SimpleBinaryKernelImpl {
  return (aShape: number[], bShape: number[], aVals: DataValues,
          bVals: DataValues, dtype: DataType): [TypedArray, number[]] => {
    const newShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);

    const resultRank = newShape.length;
    const resultStrides = util.computeStrides(newShape);
    const resultSize = util.sizeFromShape(newShape);

    const result =
        util.getTypedArrayFromDType(dtype as NumericDataType, resultSize);

    const aRank = aShape.length;
    const bRank = bShape.length;

    const aStrides = util.computeStrides(aShape);
    const bStrides = util.computeStrides(bShape);

    const aBroadcastDims = backend_util.getBroadcastDims(aShape, newShape);
    const bBroadcastDims = backend_util.getBroadcastDims(bShape, newShape);

    if (aBroadcastDims.length + bBroadcastDims.length === 0) {
      for (let i = 0; i < result.length; ++i) {
        result[i] = op(aVals[i % aVals.length], bVals[i % bVals.length]);
      }
    } else {
      for (let i = 0; i < result.length; ++i) {
        const loc = util.indexToLoc(i, resultRank, resultStrides);

        const aLoc = loc.slice(-aRank);
        aBroadcastDims.forEach(d => aLoc[d] = 0);
        const aIndex = util.locToIndex(aLoc, aRank, aStrides);

        const bLoc = loc.slice(-bRank);
        bBroadcastDims.forEach(d => bLoc[d] = 0);
        const bIndex = util.locToIndex(bLoc, bRank, bStrides);

        result[i] = op(aVals[aIndex], bVals[bIndex]);
      }
    }

    return [result, newShape];
  };
}
