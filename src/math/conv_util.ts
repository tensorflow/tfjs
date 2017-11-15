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

export type PadInfo = {
  top: number,
  left: number,
  right: number,
  bottom: number
};

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
  padInfo: PadInfo
};

export type DepthwiseConvInfo = {
  inShape: [number, number, number, number],
  outShape: [number, number, number, number],
  channelMul: number,
  strideHeight: number,
  strideWidth: number,
  filterHeight: number,
  filterWidth: number,
  padInfo: PadInfo
};

/**
 * Computes the information for a forward pass of a depthwise convolution.
 */
export function computeDepthwiseConv2DInfo(
    inShape: [number, number, number, number],
    filterShape: [number, number, number, number],
    strides: number|[number, number],
    pad: 'same'|'valid'|number): DepthwiseConvInfo {
  const [filterHeight, filterWidth, inChannels, channelMul] = filterShape;
  const [strideHeight, strideWidth] = parseTupleParam(strides);
  const inHeight = inShape[1];
  const inWidth = inShape[2];
  const batchSize = inShape[0];
  const {padInfo, outHeight, outWidth} = getPadAndOutInfo(
      pad, inHeight, inWidth, strideHeight, strideWidth, filterHeight,
      filterWidth);
  const outChannels = inChannels * channelMul;
  const outShape: [number, number, number, number] =
      [batchSize, outHeight, outWidth, outChannels];

  return {
    inShape,
    outShape,
    channelMul,
    strideHeight,
    strideWidth,
    filterHeight,
    filterWidth,
    padInfo
  };
}

/**
 * Computes the information for a forward pass of a convolution/pooling
 * operation.
 */
export function computeConv2DInfo(
    inShape: [number, number, number], filterHeight: number,
    filterWidth: number, outDepth: number, strideHeight: number,
    strideWidth: number, pad: 'same'|'valid'|number): ConvInfo {
  const inHeight = inShape[0];
  const inWidth = inShape[1];
  const {padInfo, outHeight, outWidth} = getPadAndOutInfo(
      pad, inHeight, inWidth, strideHeight, strideWidth, filterHeight,
      filterWidth);
  const outShape: [number, number, number] = [outHeight, outWidth, outDepth];
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

function parseTupleParam(param: number|[number, number]): [number, number] {
  return typeof param === 'number' ? [param, param] : param;
}

function getPadAndOutInfo(
    pad: 'same'|'valid'|number, inHeight: number, inWidth: number,
    strideHeight: number, strideWidth: number, filterHeight: number,
    filterWidth: number):
    {padInfo: PadInfo, outHeight: number, outWidth: number} {
  let padInfo: PadInfo;
  let outHeight: number;
  let outWidth: number;

  if (typeof pad === 'number') {
    padInfo = {top: pad, bottom: pad, left: pad, right: pad};
    const outShape = computeOutputShape3D(
        [inHeight, inWidth, 1], filterHeight, 1, strideHeight, pad);
    outHeight = outShape[0];
    outWidth = outShape[1];
  } else if (pad === 'same') {
    outHeight = Math.ceil(inHeight / strideHeight);
    outWidth = Math.ceil(inWidth / strideWidth);
    const padAlongHeight =
        (outHeight - 1) * strideHeight + filterHeight - inHeight;
    const padAlongWidth = (outWidth - 1) * strideWidth + filterWidth - inWidth;
    const top = Math.floor(padAlongHeight / 2);
    const bottom = padAlongHeight - top;
    const left = Math.floor(padAlongWidth / 2);
    const right = padAlongWidth - left;
    padInfo = {top, bottom, left, right};
  } else if (pad === 'valid') {
    padInfo = {top: 0, bottom: 0, left: 0, right: 0};
    outHeight = Math.ceil((inHeight - filterHeight + 1) / strideHeight);
    outWidth = Math.ceil((inWidth - filterWidth + 1) / strideWidth);
  } else {
    throw Error(`Unknown padding parameter: ${pad}`);
  }
  return {padInfo, outHeight, outWidth};
}
