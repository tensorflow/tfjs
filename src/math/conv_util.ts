/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as util from '../util';

export function computeOutputShape3D(
    inputShapeRowColDepth: [number, number, number], fieldSize: number,
    depth: number, stride: number, zeroPad?: number): [number, number, number] {
  if (zeroPad == null) {
    zeroPad = computeDefaultPad(inputShapeRowColDepth, fieldSize, stride);
  }
  const inputRows = inputShapeRowColDepth[0];
  const inputCols = inputShapeRowColDepth[1];
  const outputRows = (inputRows - fieldSize + 2 * zeroPad) / stride + 1;
  util.assert(
      util.isInt(outputRows),
      `The output # of rows (${outputRows}) must be an integer. Change the ` +
          `stride and/or zero pad parameters`);

  const outputCols = (inputCols - fieldSize + 2 * zeroPad) / stride + 1;
  util.assert(
      util.isInt(outputCols),
      `The output # of columns (${outputCols}) must be an integer. Change ` +
          `the stride and/or zero pad parameters`);

  return [outputRows, outputCols, depth];
}

export function computeDefaultPad(
    inputShape: [number, number, number], fieldSize: number,
    stride: number): number {
  return Math.floor((inputShape[0] * (stride - 1) - stride + fieldSize) / 2);
}

export function computeTexShapeFrom3D(
    shapeRowColDepth: [number, number, number]): [number, number] {
  return [shapeRowColDepth[0], shapeRowColDepth[1] * shapeRowColDepth[2]];
}

export function computeWeightsShape4D(
    inputDepth: number, outputDepth: number,
    fSize: number): [number, number, number, number] {
  return [fSize, fSize, inputDepth, outputDepth];
}

export function computeWeightsTexShape(
    inputDepth: number, outputDepth: number,
    fieldSize: number): [number, number] {
  return [fieldSize * fieldSize * inputDepth, outputDepth];
}

export function computeBiasesTexShape(outputDepth: number): [number, number] {
  return [1, outputDepth];
}

export function computeDilatedRC(
    rc: [number, number], origStride: number): [number, number] {
  const rowsDilated = (rc[0] - 1) * origStride + 1;
  const colsDilated = (rc[1] - 1) * origStride + 1;
  return [rowsDilated, colsDilated];
}
