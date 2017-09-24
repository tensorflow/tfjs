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

export function assertConcatShapesMatch(
    x1Shape: number[], x2Shape: number[], rank: number, axis: number,
    errorMessagePrefix = '') {
  util.assert(
      x1Shape.length === rank,
      errorMessagePrefix + `x1 shape should be of rank ${rank}.`);
  util.assert(
      x2Shape.length === rank,
      errorMessagePrefix + `x2 shape should be of rank ${rank}.`);

  util.assert(
      axis >= 0 && axis < rank, `axis must be between 0 and ${rank - 1}.`);

  for (let i = 0; i < rank; i++) {
    util.assert(
        (i === axis) || (x1Shape[i] === x2Shape[i]),
        errorMessagePrefix +
            `Shape (${x1Shape}) does not match (${x2Shape}) along ` +
            `the non-concatenated axis ${i}.`);
  }
}

export function computeConcatOutputShape(
    x1Shape: number[], x2Shape: number[],
    axis: number): [number, number, number] {
  util.assert(
      x1Shape.length === x2Shape.length,
      'x1 and x2 should have the same rank.');
  const outputShape = x1Shape.slice();
  outputShape[axis] += x2Shape[axis];
  return outputShape as [number, number, number];
}
