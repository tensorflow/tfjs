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

export function computeOutShape1D(
    x1Shape: number[], x2Shape: number[]): number[] {
  util.assert(
      x1Shape.length === 1 && x2Shape.length === 1,
      'x1 and x2 should be 1d array.');
  const outputShape = x1Shape.slice();
  outputShape[0] += x2Shape[0];
  return outputShape;
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

export function computeGradientSliceShapes2D(
    x1TensorShape: number[], yTensorShape: number[], axis: number): {
  x1Begin: [number, number],
  x1Size: [number, number],
  x2Begin: [number, number],
  x2Size: [number, number]
} {
  const x1AxisSize = x1TensorShape[axis];

  const x1Begin: [number, number] = [0, 0];

  const x1Size: [number, number] = [yTensorShape[0], yTensorShape[1]];
  x1Size[axis] = x1AxisSize;

  const x2Begin: [number, number] = [0, 0];
  x2Begin[axis] = x1AxisSize;

  const x2Size: [number, number] = [yTensorShape[0], yTensorShape[1]];
  x2Size[axis] = yTensorShape[axis] - x1AxisSize;

  return {x1Begin, x1Size, x2Begin, x2Size};
}

export function computeGradientSliceShapes3D(
    x1Shape: number[], yShape: number[], axis: number): {
  x1Begin: [number, number, number],
  x1Size: [number, number, number],
  x2Begin: [number, number, number],
  x2Size: [number, number, number]
} {
  const x1AxisSize = x1Shape[axis];

  const x1Begin: [number, number, number] = [0, 0, 0];

  const x1Size: [number, number, number] = [yShape[0], yShape[1], yShape[2]];
  x1Size[axis] = x1AxisSize;

  const x2Begin: [number, number, number] = [0, 0, 0];
  x2Begin[axis] = x1AxisSize;

  const x2Size: [number, number, number] = [yShape[0], yShape[1], yShape[2]];
  x2Size[axis] = yShape[axis] - x1AxisSize;

  return {x1Begin, x1Size, x2Begin, x2Size};
}

export function computeGradientSliceShapes4D(
    x1TensorShape: number[], yTensorShape: number[], axis: number): {
  x1Begin: [number, number, number, number],
  x1Size: [number, number, number, number],
  x2Begin: [number, number, number, number],
  x2Size: [number, number, number, number]
} {
  const x1AxisSize = x1TensorShape[axis];

  const x1Begin: [number, number, number, number] = [0, 0, 0, 0];

  const x1Size: [number, number, number, number] =
      [yTensorShape[0], yTensorShape[1], yTensorShape[2], yTensorShape[3]];
  x1Size[axis] = x1AxisSize;

  const x2Begin: [number, number, number, number] = [0, 0, 0, 0];
  x2Begin[axis] = x1AxisSize;

  const x2Size: [number, number, number, number] =
      [yTensorShape[0], yTensorShape[1], yTensorShape[2], yTensorShape[3]];
  x2Size[axis] = yTensorShape[axis] - x1AxisSize;

  return {x1Begin, x1Size, x2Begin, x2Size};
}
