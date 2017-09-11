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

/**
 * Information about the forward pass of a convolution/pooling operation.
 * It includes input and output shape, strides, filter size and padding
 * information.
 */
export type ConvInfo = {
  inShape: [number, number, number],
  outShape: [number, number, number],
  strideHeight: number,
  strideWidth: number,
  filterHeight: number,
  filterWidth: number,
  padInfo: {top: number, left: number, right: number, bottom: number}
};

/**
 * Computes the information about a forward pass of a convolution/pooling
 * operation.
 */
export function computeConvInfo(
    inShape: [number, number, number], filterHeight: number,
    filterWidth: number, outDepth: number, strideHeight: number,
    strideWidth: number, pad: 'same'|'valid'|number): ConvInfo {
  if (typeof pad === 'number') {
    const outShape = computeOutputShape3D(
        inShape, filterHeight, outDepth, strideHeight, pad);
    return {
      inShape,
      outShape,
      padInfo: {top: pad, bottom: pad, left: pad, right: pad},
      strideHeight,
      strideWidth,
      filterHeight,
      filterWidth
    };
  }
  const inHeight = inShape[0];
  const inWidth = inShape[1];
  let outShape: [number, number, number];
  let padInfo: {left: number, top: number, bottom: number, right: number};
  if (pad === 'same') {
    const outHeight = Math.ceil(inHeight / strideHeight);
    const outWidth = Math.ceil(inWidth / strideWidth);
    outShape = [outHeight, outWidth, outDepth];
    const padAlongHeight =
        (outHeight - 1) * strideHeight + filterHeight - inHeight;
    const padAlongWidth = (outWidth - 1) * strideWidth + filterWidth - inWidth;
    const top = Math.floor(padAlongHeight / 2);
    const bottom = padAlongHeight - top;
    const left = Math.floor(padAlongWidth / 2);
    const right = padAlongWidth - left;
    padInfo = {top, bottom, left, right};
  } else if (pad === 'valid') {
    const outHeight = Math.ceil((inHeight - filterHeight + 1) / strideHeight);
    const outWidth = Math.ceil((inWidth - filterWidth + 1) / strideWidth);
    outShape = [outHeight, outWidth, outDepth];
    padInfo = {top: 0, bottom: 0, left: 0, right: 0};
  } else {
    throw Error(`Unknown padding parameter: ${pad}`);
  }
  return {
    inShape,
    outShape,
    padInfo,
    strideHeight,
    strideWidth,
    filterHeight,
    filterWidth
  };
}

/**
 * @deprecated Use `conv_util.computeConvInfo` instead.
 */
export function computeOutputShape3D(
    inShape: [number, number, number], fieldSize: number, outDepth: number,
    stride: number, zeroPad?: number): [number, number, number] {
  if (zeroPad == null) {
    zeroPad = computeDefaultPad(inShape, fieldSize, stride);
  }
  const inputRows = inShape[0];
  const inputCols = inShape[1];
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

  return [outputRows, outputCols, outDepth];
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
    inputDepth: number, outputDepth: number, filterHeight: number,
    filterWidth: number): [number, number, number, number] {
  return [filterHeight, filterWidth, inputDepth, outputDepth];
}

export function computeDilatedRC(
    rc: [number, number], origStride: number): [number, number] {
  const rowsDilated = (rc[0] - 1) * origStride + 1;
  const colsDilated = (rc[1] - 1) * origStride + 1;
  return [rowsDilated, colsDilated];
}
