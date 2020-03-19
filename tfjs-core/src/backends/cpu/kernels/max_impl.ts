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

import {DataType, NumericDataType, TypedArray} from '../../../types';
import * as util from '../../../util';

export const max =
    (x: TypedArray, reduceSize: number, outShape: number[], dtype: DataType):
        TypedArray => {
          const outValues = util.getTypedArrayFromDType(
              dtype as NumericDataType, util.sizeFromShape(outShape));

          for (let i = 0; i < x.length; ++i) {
            const offset = i * reduceSize;
            let max = x[offset];
            for (let j = 0; j < reduceSize; ++j) {
              const value = x[offset + j];
              if (value > max) {
                max = value;
              }
            }
            outValues[i] = max;
          }

          return outValues;
        };
