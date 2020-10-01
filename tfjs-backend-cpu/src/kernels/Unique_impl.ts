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

import {BackendValues, DataType, TypedArray, util} from '@tensorflow/tfjs-core';

export function uniqueImpl(values: BackendValues, dtype: DataType):
    {outputValues: BackendValues, indices: BackendValues} {
  let xValues: TypedArray|string[] = [];
  if (dtype === 'string') {
    xValues = (values as Uint8Array[]).map(d => util.decodeString(d));
  } else {
    xValues = values as TypedArray;
  }

  // A map from unique value to its index in outputValues.
  const uniqueValues = new Map<number|string, number>();
  const outputValues = [];
  const indices = new Int32Array(xValues.length);
  for (let i = 0; i < xValues.length; i++) {
    const value = xValues[i];
    if (uniqueValues.has(value)) {
      indices[i] = uniqueValues.get(value);
    } else {
      const uniqueIndex = uniqueValues.size;
      uniqueValues.set(value, uniqueIndex);
      indices[i] = uniqueIndex;
      outputValues.push(values[i]);
    }
  }

  return {
    outputValues: outputValues as BackendValues,
    indices,
  };
}
