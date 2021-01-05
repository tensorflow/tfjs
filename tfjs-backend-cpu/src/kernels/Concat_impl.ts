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

import {backend_util, BackendValues, DataType, TypedArray, util} from '@tensorflow/tfjs-core';

export function concatImpl(
    inputs: Array<{vals: BackendValues, shape: number[]}>, outShape: number[],
    dtype: DataType, simplyConcat: boolean): TypedArray|string[] {
  const outVals = util.getArrayFromDType(dtype, util.sizeFromShape(outShape));

  if (simplyConcat && dtype !== 'string') {
    // Use built-in TypedArray.set() method for speed.
    let offset = 0;
    inputs.forEach(input => {
      const size = util.sizeFromShape(input.shape);

      (outVals as TypedArray).set(input.vals as TypedArray, offset);
      offset += size;
    });
  } else {
    let colOffset = 0;

    inputs.forEach(input => {
      const decodedData = dtype === 'string' ?
          backend_util.fromUint8ToStringArray(input.vals as Uint8Array[]) :
          input.vals as TypedArray;

      let tIdx = 0;

      for (let row = 0; row < input.shape[0]; ++row) {
        const resIdx = row * outShape[1] + colOffset;
        for (let col = 0; col < input.shape[1]; ++col) {
          outVals[resIdx + col] = decodedData[tIdx++];
        }
      }

      colOffset += input.shape[1];
    });
  }

  return outVals;
}
