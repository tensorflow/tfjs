/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

/** An implementation of the inTopK kernel shared between webgl and cpu. */

import {tensor} from '../ops/tensor_ops';
import {Tensor} from '../tensor';
import {TypedArray} from '../types';
import {getTypedArrayFromDType} from '../util';

export function inTopKImpl<T extends Tensor>(
  predictionsVals: TypedArray, predictionsShape: number[],
  targetsVals: TypedArray, targetsShape: number[], k: number
): T {
  // Reshape predictionsVals into a 2d tensor [batch, lastDim]
  // and look up topK along lastDim.
  const lastDim = predictionsShape[predictionsShape.length - 1];
  const [batch, size] = [predictionsVals.length / lastDim, lastDim];
  const precision = getTypedArrayFromDType('bool', batch);

  for (let b = 0; b < batch; b++) {
    const offset = b * size;
    const vals = predictionsVals.subarray(offset, offset + size);
    const valAndInd: Array<{ value: number, index: number }> = [];
    for (let i = 0; i < vals.length; i++) {
      valAndInd.push({value: vals[i], index: i});
    }
    valAndInd.sort((a, b) => b.value - a.value);

    precision[b] = 0;
    for (let i = 0; i < k; i++) {
      if (valAndInd[i].index === targetsVals[b]) {
        precision[b] = 1;
        break;
      }
    }
  }

  // Output precision has the same shape as targets.
  return tensor(precision, targetsShape, 'bool') as T;
}