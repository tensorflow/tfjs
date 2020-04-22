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

import {DataType, NumericDataType, TensorInfo, TypedArray, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {reduce} from '../kernel_utils/reduce';
import {reshape} from '../kernel_utils/reshape';

export const maxImpl =
    (x: TensorInfo, reduceShape: number[], outShape: number[],
     backend: MathBackendWebGL): TensorInfo => {
      const inSize = util.sizeFromShape(reduceShape);
      const xSize = util.sizeFromShape(x.shape);
      const batchSize = xSize / inSize;

      return reshape(
          reduce(
              reshape(x, [batchSize, inSize], backend), x.dtype, 'max',
              backend),
          outShape, backend);
    };

// todo(@annxingyuan) import this from cpu backend.
export function maxImplCPU(
    aVals: TypedArray, reduceSize: number, outShape: number[],
    dtype: DataType): TypedArray {
  const vals = util.getTypedArrayFromDType(
      dtype as NumericDataType, util.sizeFromShape(outShape));

  for (let i = 0; i < vals.length; ++i) {
    const offset = i * reduceSize;
    let max = aVals[offset];
    for (let j = 0; j < reduceSize; ++j) {
      const value = aVals[offset + j];
      if (value > max) {
        max = value;
      }
    }
    vals[i] = max;
  }
  return vals;
}
