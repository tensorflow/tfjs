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
  util.assert(x1Shape.length === 1 && x2Shape.length === 1,
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

function create2DTuple(v1: number, v2: number)
    : [number, number] {
  return [v1, v2];
}

function create3DTuple(v1: number, v2: number, v3: number)
    : [number, number, number] {
  return [v1, v2, v3];
}

function create4DTuple(v1: number, v2: number, v3: number, v4: number)
    : [number, number, number, number] {
  return [v1, v2, v3, v4];
}

export function computeSliceSizes2D(x1TensorShape: number[], 
    yTensorShape: number[], axis: number) {
 util.assert(
    x1TensorShape.length === 2,
    'Input should be 2D rank.');
 util.assert(
    yTensorShape.length === 2,
    'Output should be 2D rank.'); 
 const x1AxisSize = x1TensorShape[axis];

 const x1Begin: [number, number] = create2DTuple(0, 0);
 const x1Size : [number, number] 
   = create2DTuple(yTensorShape[0], yTensorShape[1]);
 x1Size[axis] = x1AxisSize;
 const x2Begin: [number, number] = create2DTuple(0, 0);
 const x2Size : [number, number] 
   = create2DTuple(yTensorShape[0], yTensorShape[1]);
 x2Begin[axis] = x1AxisSize;
 x2Size[axis] = yTensorShape[axis] - x1AxisSize;

 return {"x1Begin": x1Begin, "x1Size": x1Size, 
   "x2Begin": x2Begin, "x2Size": x2Size};
}

export function computeSliceSizes3D(x1TensorShape: number[], 
    yTensorShape: number[], axis: number) {
 util.assert(
    x1TensorShape.length === 3,
    'Input should be 3D rank.');
 util.assert(
    yTensorShape.length === 3,
    'Output should be 3D rank.');
 const x1AxisSize = x1TensorShape[axis];

 const x1Begin: [number, number, number] = create3DTuple(0, 0, 0);
 const x1Size : [number, number, number] 
   = create3DTuple(yTensorShape[0], yTensorShape[1], yTensorShape[2]);
 x1Size[axis] = x1AxisSize;
 const x2Begin: [number, number, number] = create3DTuple(0, 0, 0);
 const x2Size : [number, number, number] 
   = create3DTuple(yTensorShape[0], yTensorShape[1], yTensorShape[2]);
 x2Begin[axis] = x1AxisSize;
 x2Size[axis] = yTensorShape[axis] - x1AxisSize;

 return {"x1Begin": x1Begin, "x1Size": x1Size, 
   "x2Begin": x2Begin, "x2Size": x2Size};
}

export function computeSliceSizes4D(x1TensorShape: number[], 
    yTensorShape: number[], axis: number) {
 util.assert(
    x1TensorShape.length === 4,
    'Input should be 4D rank.');
 util.assert(
    yTensorShape.length === 4,
    'Output should be 4D rank.');
 const x1AxisSize = x1TensorShape[axis];

 const x1Begin: [number, number, number, number] = create4DTuple(0, 0, 0, 0);
 const x1Size : [number, number, number, number] 
   = create4DTuple(yTensorShape[0], yTensorShape[1], 
         yTensorShape[2], yTensorShape[3]);
 x1Size[axis] = x1AxisSize;
 const x2Begin: [number, number, number, number] = create4DTuple(0, 0, 0, 0);
 const x2Size : [number, number, number, number] 
   = create4DTuple(yTensorShape[0], yTensorShape[1], 
        yTensorShape[2], yTensorShape[3]);
 x2Begin[axis] = x1AxisSize;
 x2Size[axis] = yTensorShape[axis] - x1AxisSize;

 return {"x1Begin": x1Begin, "x1Size": x1Size, 
   "x2Begin": x2Begin, "x2Size": x2Size};
}
