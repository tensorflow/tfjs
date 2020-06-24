/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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

/**
 * Returns true if the axis specifies the inner most dimensions of the
 * array.
 */
export function axesAreInnerMostDims(axes: number[], rank: number): boolean {
  for (let i = 0; i < axes.length; ++i) {
    if (axes[axes.length - i - 1] !== rank - 1 - i) {
      return false;
    }
  }
  return true;
}

export function combineLocations(
    outputLoc: number[], reduceLoc: number[], axes: number[]): number[] {
  const rank = outputLoc.length + reduceLoc.length;
  const loc = [];
  let outIdx = 0;
  let reduceIdx = 0;
    for (let dim = 0; dim < rank; dim++) {
    if (axes.indexOf(dim) === -1) {
      loc.push(outputLoc[outIdx++]);
    } else {
      loc.push(reduceLoc[reduceIdx++]);
    }
  }
  return loc;
}

export function computeOutAndReduceShapes(
    aShape: number[], axes: number[]): [number[], number[]] {
  const outShape = [];
  const rank = aShape.length;
  for (let dim = 0; dim < rank; dim++) {
    if (axes.indexOf(dim) === -1) {
      outShape.push(aShape[dim]);
    }
  }
  const reduceShape = axes.map(dim => aShape[dim]);
  return [outShape, reduceShape];
}

export function expandShapeToKeepDim(
    shape: number[], axes: number[]): number[] {
  const reduceSubShape = axes.map(x => 1);
  return combineLocations(shape, reduceSubShape, axes);
}

export function assertAxesAreInnerMostDims(
    msg: string, axes: number[], rank: number): void {
  util.assert(
      axesAreInnerMostDims(axes, rank),
      () => `${msg} supports only inner-most axes for now. ` +
          `Got axes ${axes} and rank-${rank} input.`);
}

/**
 * Returns the axes permutation to be used with `tf.transpose`, if such
 * permutation is necessary. Otherwise it returns null. This method is used by
 * operations that operate only on inner-most axes.
 */
export function getAxesPermutation(axes: number[], rank: number): number[]|
    null {
  if (axesAreInnerMostDims(axes, rank)) {
    return null;
  }
  const result: number[] = [];
  for (let i = 0; i < rank; ++i) {
    if (axes.indexOf(i) === -1) {
      result.push(i);
    }
  }
  axes.forEach(axis => result.push(axis));
  return result;
}

/** Returns the axes permutation that undoes the original permutation. */
export function getUndoAxesPermutation(axes: number[]): number[] {
  return axes.map((axis, i) => [i, axis])
      .sort((a, b) => a[1] - b[1])
      .map(x => x[0]);
}

export function getInnerMostAxes(numAxes: number, rank: number): number[] {
  const res: number[] = [];
  for (let i = rank - numAxes; i < rank; ++i) {
    res.push(i);
  }
  return res;
}
