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

export function assertParams(aShape: number[], bShape: number[], axis: number) {
  const aRank = aShape.length;
  const bRank = bShape.length;
  util.assert(
      aShape.length === bShape.length,
      `Error in concat${aRank}D: rank of x1 (${aRank}) and x2 (${bRank}) ` +
          `must be the same.`);

  util.assert(
      axis >= 0 && axis < aRank,
      `Error in concat${aRank}D: axis must be ` +
          `between 0 and ${aRank - 1}.`);

  for (let i = 0; i < aRank; i++) {
    util.assert(
        (i === axis) || (aShape[i] === bShape[i]),
        `Error in concat${aRank}D: Shape (${aShape}) does not match ` +
            `(${bShape}) along the non-concatenated axis ${i}.`);
  }
}

export function computeOutShape(
    x1Shape: number[], x2Shape: number[], axis: number): number[] {
  util.assert(
      x1Shape.length === x2Shape.length,
      'x1 and x2 should have the same rank.');
  const outputShape = x1Shape.slice();
  outputShape[axis] += x2Shape[axis];
  return outputShape;
}

export function computeGradientSliceShapes(
    aShape: [number, number], bShape: [number, number]) {
  return {
    aBegin: [0, 0] as [number, number],
    aSize: aShape,
    bBegin: [0, aShape[1]] as [number, number],
    bSize: bShape
  };
}
