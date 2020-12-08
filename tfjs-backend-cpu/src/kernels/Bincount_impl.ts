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

import {buffer, DataType, Rank, TensorBuffer, TypedArray, util} from '@tensorflow/tfjs-core';

export function bincountImpl(
    xVals: TypedArray, weightsVals: TypedArray, weightsDtype: DataType,
    weightsShape: number[], size: number): TypedArray {
  const weightsSize = util.sizeFromShape(weightsShape);
  const outVals = util.makeZerosTypedArray(size, weightsDtype) as TypedArray;

  for (let i = 0; i < xVals.length; i++) {
    const value = xVals[i];
    if (value < 0) {
      throw new Error('Input x must be non-negative!');
    }

    if (value >= size) {
      continue;
    }

    if (weightsSize > 0) {
      // weights can be any size, for out of range reference, set weights
      // to be zero.
      const weightsVal = i < weightsVals.length ? weightsVals[i] : 0;
      outVals[value] += weightsVal || 0;
    } else {
      outVals[value] += 1;
    }
  }

  return outVals;
}

export function bincountReduceImpl<R extends Rank>(
    xBuf: TensorBuffer<R>, weightsBuf: TensorBuffer<R>, size: number,
    binaryOutput = false): TensorBuffer<R> {
  const numRows = xBuf.shape[0];
  const numCols = xBuf.shape[1];

  const outBuf = buffer([numRows, size], weightsBuf.dtype);

  for (let i = 0; i < numRows; i++) {
    for (let j = 0; j < numCols; j++) {
      const value = xBuf.get(i, j);
      if (value < 0) {
        throw new Error('Input x must be non-negative!');
      }

      if (value >= size) {
        continue;
      }

      if (binaryOutput) {
        outBuf.set(1, i, value);
      } else {
        if (weightsBuf.size > 0) {
          // The behavior here is different than bincountImpl above. We don't
          // set weight to be zero for out of range reference. This is because,
          // in TF c++ implementation, it does not handle this case explicitly.
          // TF will output random numbers. Instead, TFJS will throw error.
          // If TF changes implementation to default to zero, we should make
          // that change too.
          outBuf.set(outBuf.get(i, value) + weightsBuf.get(i, j), i, value);
        } else {
          outBuf.set(outBuf.get(i, value) + 1, i, value);
        }
      }
    }
  }

  return outBuf as TensorBuffer<R>;
}
