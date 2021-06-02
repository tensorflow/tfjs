/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {TypedArray, util} from '@tensorflow/tfjs-core';

function split(
    str: Uint8Array, delimiters: Uint8Array, skipEmpty: boolean): Uint8Array[] {
  if (!str.length) {
    return [];
  }
  // When the delimiter is empty, the input is split into individual characters.
  if (delimiters.length === 0) {
    const result: Uint8Array[] = new Array(str.length);
    for (let i = 0; i < str.length; ++i) {
      result[i] = str.subarray(i, i + 1);
    }
    return result;
  }
  // When there is one delimiter, the input is split only at that delimiter.
  if (delimiters.length === 1) {
    const delimiter = delimiters[0];
    const result: Uint8Array[] = [];
    let f = str.indexOf(delimiter);
    while (f !== -1) {
      const token = str.subarray(0, f);
      if (!skipEmpty || token.length !== 0) {
        result.push(token);
      }
      str = str.subarray(f + 1);
      f = str.indexOf(delimiter);
    }
    if (!skipEmpty || str.length !== 0) {
      result.push(str);
    }
    return result;
  }
  // When there are multiple delimiters, the input is split at every instance
  // one of the delimiters appears.
  const result: Uint8Array[] = [];
  let tokenStart = 0;
  for (let i = 0; i < str.length + 1; i++) {
    if ((i === str.length) || (delimiters.indexOf(str[i]) !== -1)) {
      const token = str.subarray(tokenStart, i);
      if (!skipEmpty || token.length !== 0) {
        result.push(token);
      }
      tokenStart = i + 1;
    }
  }
  return result;
}

export function stringSplitImpl(
    input: Uint8Array[], delimiter: Uint8Array,
    skipEmpty: boolean): [TypedArray, Uint8Array[], [number, number]] {
  const batchSize = input.length;

  // Empty delimiter means split the input character by character.
  const tokens: Uint8Array[] = [];

  let outputSize = 0;
  let maxNumEntries = 0;
  const numIndices: number[] = new Array(batchSize);
  for (let i = 0; i < batchSize; ++i) {
    const parts = split(input[i], delimiter, skipEmpty);
    const nEntries = parts.length;
    numIndices[i] = nEntries;
    outputSize += nEntries;
    maxNumEntries = Math.max(maxNumEntries, nEntries);
    tokens.push(...parts);
  }

  const indices = util.getArrayFromDType('int32', outputSize * 2) as TypedArray;
  const values: Uint8Array[] = new Array(outputSize);
  const shape: [number, number] = [batchSize, maxNumEntries];

  let c = 0;
  for (let i = 0; i < batchSize; ++i) {
    for (let j = 0; j < numIndices[i]; ++j) {
      // indices is a 2d tensor with shape of [outputSize, 2]
      indices[c * 2] = i;
      indices[c * 2 + 1] = j;
      values[c] = tokens[c];
      ++c;
    }
  }

  return [indices, values, shape];
}
