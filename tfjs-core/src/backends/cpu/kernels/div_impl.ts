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

import * as broadcast_util from '../../../ops/broadcast_util';
import {TypedArray} from '../../../types';
import * as util from '../../../util';

export const div =
    (a: TypedArray, aShape: number[], b: TypedArray, bShape: number[],
     outValues: TypedArray, outShape: number[]): TypedArray => {
      const aBroadcastDims = broadcast_util.getBroadcastDims(aShape, outShape);
      const bBroadcastDims = broadcast_util.getBroadcastDims(bShape, outShape);

      const aStrides = util.computeStrides(aShape);
      const bStrides = util.computeStrides(bShape);
      const outStrides = util.computeStrides(outShape);

      if (aBroadcastDims.length + bBroadcastDims.length === 0) {
        for (let i = 0; i < outValues.length; ++i) {
          outValues[i] = a[i % a.length] / b[i % b.length];
        }
      } else {
        for (let i = 0; i < outValues.length; ++i) {
          const loc = util.indexToLoc(i, outStrides);

          const aLoc = loc.slice(-aShape.length);
          aBroadcastDims.forEach(d => aLoc[d] = 0);
          const aIndex = util.locToIndex(aLoc, aStrides);

          const bLoc = loc.slice(-bShape.length);
          bBroadcastDims.forEach(d => bLoc[d] = 0);
          const bIndex = util.locToIndex(bLoc, bStrides);

          outValues[i] = a[aIndex] / b[bIndex];
        }
      }
      return outValues;
    };
