/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

import * as util from '../util';

import {NDArray} from './ndarray';

export function assertParamsValid(
    input: NDArray, begin: number[], size: number[]): void {
  util.assert(
      input.rank === begin.length,
      `Error in slice${input.rank}D: Length of begin ${begin} must ` +
          `match the rank of the array (${input.rank}).`);
  util.assert(
      input.rank === size.length,
      `Error in slice${input.rank}D: Length of size ${size} must ` +
          `match the rank of the array (${input.rank}).`);

  for (let i = 0; i < input.rank; ++i) {
    util.assert(
        begin[i] + size[i] <= input.shape[i],
        `Error in slice${input.rank}D: begin[${i}] + size[${i}] ` +
            `(${begin[i] + size[i]}) would overflow input.shape[${i}] (${
                input.shape[i]})`);
  }
}
