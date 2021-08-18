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

import {DataTypeMap, util} from '@tensorflow/tfjs-core';

export function rangeImpl(
    start: number, stop: number, step: number,
    dtype: 'float32'|'int32'): DataTypeMap['float32' | 'int32'] {
  const sameStartStop = start === stop;
  const increasingRangeNegativeStep = start < stop && step < 0;
  const decreasingRangePositiveStep = stop < start && step > 1;

  if (sameStartStop || increasingRangeNegativeStep ||
      decreasingRangePositiveStep) {
    return util.makeZerosTypedArray(0, dtype);
  }

  const numElements = Math.abs(Math.ceil((stop - start) / step));
  const values = util.makeZerosTypedArray(numElements, dtype);

  if (stop < start && step === 1) {
    // Auto adjust the step's sign if it hasn't been set
    // (or was set to 1)
    step = -1;
  }

  values[0] = start;
  for (let i = 1; i < values.length; i++) {
    values[i] = values[i - 1] + step;
  }
  return values;
}
