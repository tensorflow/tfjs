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

type PadType = 'SAME'|'VALID'|'NUMBER'|'EXPLICIT';

// For NHWC should be in the following form:
//  [[0, 0], [pad_top,pad_bottom], [pad_left, pad_right], [0, 0]]
// For NCHW should be in the following form:
//  [[0, 0], [0, 0], [pad_top,pad_bottom], [pad_left, pad_right]]
// Reference: https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
export type ExplicitPadding =
    [[number, number], [number, number], [number, number], [number, number]];

export type PadInfo = {
  top: number,
  left: number,
  right: number,
  bottom: number,
  type: PadType
};

export type PadInfo3D = {
  top: number,
  left: number,
  right: number,
  bottom: number,
  front: number,
  back: number,
  type: PadType
};

/**
 * Information about the forward pass of a convolution/pooling operation.
 * It includes input and output shape, strides, filter size and padding
 * information.
 */
export type Conv2DInfo = {
  batchSize: number,
  inHeight: number,
  inWidth: number,
  inChannels: number,
  outHeight: number,
  outWidth: number,
  outChannels: number,
  dataFormat: 'channelsFirst'|'channelsLast',
  strideHeight: number,
  strideWidth: number,
  dilationHeight: number,
  dilationWidth: number,
  filterHeight: number,
  filterWidth: number,
  effectiveFilterHeight: number,
  effectiveFilterWidth: number,
  padInfo: PadInfo,
  inShape: [number, number, number, number],
  outShape: [number, number, number, number],
  filterShape: [number, number, number, number]
};

export function computePool2DInfo(
    inShape: [number, number, number, number],
    filterSize: [number, number]|number, strides: number|[number, number],
    dilations: number|[number, number], pad: 'same'|'valid'|number,
    roundingMode?: 'floor'|'round'|'ceil',
    dataFormat: 'channelsFirst'|'channelsLast' = 'channelsLast'): Conv2DInfo {
  const [filterHeight, filterWidth] = parseTupleParam(filterSize);

  let filterShape: [number, number, number, number];
  if (dataFormat === 'channelsLast') {
    filterShape = [filterHeight, filterWidth, inShape[3], inShape[3]];
  } else if (dataFormat === 'channelsFirst') {
    filterShape = [filterHeight, filterWidth, inShape[1], inShape[1]];
  } else {
    throw new Error(`Unknown dataFormat ${dataFormat}`);
  }

  return computeConv2DInfo(
      inShape, filterShape, strides, dilations, pad, roundingMode, false,
      dataFormat);
}

/**
 * Computes the information for a forward pass of a pooling3D operation.
 */
export function computePool3DInfo(
    inShape: [number, number, number, number, number],
    filterSize: number|[number, number, number],
    strides: number|[number, number, number],
    dilations: number|[number, number, number], pad: 'same'|'valid'|number,
    roundingMode?: 'floor'|'round'|'ceil',
    dataFormat: 'NDHWC'|'NCDHW' = 'NDHWC'): Conv3DInfo {
  const [filterDepth, filterHeight, filterWidth] = parse3TupleParam(filterSize);

  let filterShape: [number, number, number, number, number];
  let $dataFormat: 'channelsFirst'|'channelsLast';
  if (dataFormat === 'NDHWC') {
    $dataFormat = 'channelsLast';
    filterShape =
        [filterDepth, filterHeight, filterWidth, inShape[4], inShape[4]];
  } else if (dataFormat === 'NCDHW') {
    $dataFormat = 'channelsFirst';
    filterShape =
        [filterDepth, filterHeight, filterWidth, inShape[1], inShape[1]];
  } else {
    throw new Error(`Unknown dataFormat ${dataFormat}`);
  }

  return computeConv3DInfo(
      inShape, filterShape, strides, dilations, pad, false, $dataFormat,
      roundingMode);
}

/**
 * Computes the information for a forward pass of a convolution/pooling
 * operation.
 */
export function computeConv2DInfo(
    inShape: [number, number, number, number],
    filterShape: [number, number, number, number],
    strides: number|[number, number], dilations: number|[number, number],
    pad: 'same'|'valid'|number|ExplicitPadding,
    roundingMode?: 'floor'|'round'|'ceil', depthwise = false,
    dataFormat: 'channelsFirst'|'channelsLast' = 'channelsLast'): Conv2DInfo {
  let [batchSize, inHeight, inWidth, inChannels] = [-1, -1, -1, -1];
  if (dataFormat === 'channelsLast') {
    [batchSize, inHeight, inWidth, inChannels] = inShape;
  } else if (dataFormat === 'channelsFirst') {
    [batchSize, inChannels, inHeight, inWidth] = inShape;
  } else {
    throw new Error(`Unknown dataFormat ${dataFormat}`);
  }

  const [filterHeight, filterWidth, , filterChannels] = filterShape;
  const [strideHeight, strideWidth] = parseTupleParam(strides);
  const [dilationHeight, dilationWidth] = parseTupleParam(dilations);

  const effectiveFilterHeight =
      getEffectiveFilterSize(filterHeight, dilationHeight);
  const effectiveFilterWidth =
      getEffectiveFilterSize(filterWidth, dilationWidth);
  const {padInfo, outHeight, outWidth} = getPadAndOutInfo(
      pad, inHeight, inWidth, strideHeight, strideWidth, effectiveFilterHeight,
      effectiveFilterWidth, roundingMode, dataFormat);

  const outChannels = depthwise ? filterChannels * inChannels : filterChannels;

  let outShape: [number, number, number, number];
  if (dataFormat === 'channelsFirst') {
    outShape = [batchSize, outChannels, outHeight, outWidth];
  } else if (dataFormat === 'channelsLast') {
    outShape = [batchSize, outHeight, outWidth, outChannels];
  }

  return {
    batchSize,
    dataFormat,
    inHeight,
    inWidth,
    inChannels,
    outHeight,
    outWidth,
    outChannels,
    padInfo,
    strideHeight,
    strideWidth,
    filterHeight,
    filterWidth,
    effectiveFilterHeight,
    effectiveFilterWidth,
    dilationHeight,
    dilationWidth,
    inShape,
    outShape,
    filterShape
  };
}

/**
 * Information about the forward pass of a 3D convolution/pooling operation.
 * It includes input and output shape, strides, filter size and padding
 * information.
 */
export type Conv3DInfo = {
  batchSize: number,
  inDepth: number,
  inHeight: number,
  inWidth: number,
  inChannels: number,
  outDepth: number,
  outHeight: number,
  outWidth: number,
  outChannels: number,
  dataFormat: 'channelsFirst'|'channelsLast',
  strideDepth: number,
  strideHeight: number,
  strideWidth: number,
  dilationDepth: number,
  dilationHeight: number,
  dilationWidth: number,
  filterDepth: number,
  filterHeight: number,
  filterWidth: number,
  effectiveFilterDepth: number,
  effectiveFilterHeight: number,
  effectiveFilterWidth: number,
  padInfo: PadInfo3D,
  inShape: [number, number, number, number, number],
  outShape: [number, number, number, number, number],
  filterShape: [number, number, number, number, number]
};

/**
 * Computes the information for a forward pass of a 3D convolution/pooling
 * operation.
 */
export function computeConv3DInfo(
    inShape: [number, number, number, number, number],
    filterShape: [number, number, number, number, number],
    strides: number|[number, number, number],
    dilations: number|[number, number, number], pad: 'same'|'valid'|number,
    depthwise = false,
    dataFormat: 'channelsFirst'|'channelsLast' = 'channelsLast',
    roundingMode?: 'floor'|'round'|'ceil'): Conv3DInfo {
  let [batchSize, inDepth, inHeight, inWidth, inChannels] =
      [-1, -1, -1, -1, -1];
  if (dataFormat === 'channelsLast') {
    [batchSize, inDepth, inHeight, inWidth, inChannels] = inShape;
  } else if (dataFormat === 'channelsFirst') {
    [batchSize, inChannels, inDepth, inHeight, inWidth] = inShape;
  } else {
    throw new Error(`Unknown dataFormat ${dataFormat}`);
  }

  const [filterDepth, filterHeight, filterWidth, , filterChannels] =
      filterShape;
  const [strideDepth, strideHeight, strideWidth] = parse3TupleParam(strides);
  const [dilationDepth, dilationHeight, dilationWidth] =
      parse3TupleParam(dilations);

  const effectiveFilterDepth =
      getEffectiveFilterSize(filterDepth, dilationDepth);
  const effectiveFilterHeight =
      getEffectiveFilterSize(filterHeight, dilationHeight);
  const effectiveFilterWidth =
      getEffectiveFilterSize(filterWidth, dilationWidth);
  const {padInfo, outDepth, outHeight, outWidth} = get3DPadAndOutInfo(
      pad, inDepth, inHeight, inWidth, strideDepth, strideHeight, strideWidth,
      effectiveFilterDepth, effectiveFilterHeight, effectiveFilterWidth,
      roundingMode);

  const outChannels = depthwise ? filterChannels * inChannels : filterChannels;

  let outShape: [number, number, number, number, number];
  if (dataFormat === 'channelsFirst') {
    outShape = [batchSize, outChannels, outDepth, outHeight, outWidth];
  } else if (dataFormat === 'channelsLast') {
    outShape = [batchSize, outDepth, outHeight, outWidth, outChannels];
  }

  return {
    batchSize,
    dataFormat,
    inDepth,
    inHeight,
    inWidth,
    inChannels,
    outDepth,
    outHeight,
    outWidth,
    outChannels,
    padInfo,
    strideDepth,
    strideHeight,
    strideWidth,
    filterDepth,
    filterHeight,
    filterWidth,
    effectiveFilterDepth,
    effectiveFilterHeight,
    effectiveFilterWidth,
    dilationDepth,
    dilationHeight,
    dilationWidth,
    inShape,
    outShape,
    filterShape
  };
}

function computeOutputShape2D(
    inShape: [number, number], fieldSize: number, stride: number,
    zeroPad?: number, roundingMode?: 'floor'|'round'|'ceil'): [number, number] {
  if (zeroPad == null) {
    zeroPad = computeDefaultPad(inShape, fieldSize, stride);
  }
  const inputRows = inShape[0];
  const inputCols = inShape[1];

  const outputRows = conditionalRound(
      (inputRows - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
  util.assert(
      util.isInt(outputRows),
      () => `The output # of rows (${outputRows}) must be an integer. ` +
          `Change the stride and/or zero pad parameters`);

  const outputCols = conditionalRound(
      (inputCols - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
  util.assert(
      util.isInt(outputCols),
      () => `The output # of columns (${outputCols}) must be an integer. ` +
          `Change the stride and/or zero pad parameters`);

  return [outputRows, outputCols];
}

function computeOutputShape4D(
    inShape: [number, number, number, number], fieldSize: number,
    outChannels: number, stride: number, zeroPad?: number,
    roundingMode?: 'floor'|'round'|'ceil'): [number, number, number, number] {
  if (zeroPad == null) {
    zeroPad = computeDefaultPad(inShape, fieldSize, stride);
  }
  const inputDepth = inShape[0];
  const inputRows = inShape[1];
  const inputCols = inShape[2];

  const outputDepths = conditionalRound(
      (inputDepth - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
  util.assert(
      util.isInt(outputDepths),
      () => `The output # of depths (${outputDepths}) must be an integer. ` +
          `Change the stride and/or zero pad parameters`);

  const outputRows = conditionalRound(
      (inputRows - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
  util.assert(
      util.isInt(outputRows),
      () => `The output # of rows (${outputRows}) must be an integer. ` +
          `Change the stride and/or zero pad parameters`);

  const outputCols = conditionalRound(
      (inputCols - fieldSize + 2 * zeroPad) / stride + 1, roundingMode);
  util.assert(
      util.isInt(outputCols),
      () => `The output # of columns (${outputCols}) must be an integer. ` +
          `Change the stride and/or zero pad parameters`);

  return [outputDepths, outputRows, outputCols, outChannels];
}

export function computeDefaultPad(
    inputShape: [number, number]|[number, number, number, number],
    fieldSize: number, stride: number, dilation = 1): number {
  const effectiveFieldSize = getEffectiveFilterSize(fieldSize, dilation);
  return Math.floor(
      (inputShape[0] * (stride - 1) - stride + effectiveFieldSize) / 2);
}

function parseTupleParam(param: number|number[]): [number, number, number] {
  if (typeof param === 'number') {
    return [param, param, param];
  }
  if (param.length === 2) {
    return [param[0], param[1], 1];
  }
  return param as [number, number, number];
}

function parse3TupleParam(param: number|[number, number, number]):
    [number, number, number] {
  return typeof param === 'number' ? [param, param, param] : param;
}

/* See https://www.tensorflow.org/api_docs/python/tf/nn/atrous_conv2d
 * Atrous convolution is equivalent to standard convolution with upsampled
 * filters with effective_filter_height =
 * filter_height + (filter_height - 1) * (dilation - 1)
 * and effective_filter_width =
 * filter_width + (filter_width - 1) * (dilation - 1),
 * produced by inserting dilation - 1 zeros along consecutive elements across
 * the filters' spatial dimensions.
 * When there is a dilation, this converts a filter dimension to the
 * effective filter dimension, so it can be used in a standard convolution.
 */
function getEffectiveFilterSize(filterSize: number, dilation: number) {
  if (dilation <= 1) {
    return filterSize;
  }

  return filterSize + (filterSize - 1) * (dilation - 1);
}

function getPadAndOutInfo(
    pad: 'same'|'valid'|number|ExplicitPadding, inHeight: number,
    inWidth: number, strideHeight: number, strideWidth: number,
    filterHeight: number, filterWidth: number,
    roundingMode: 'floor'|'round'|'ceil',
    dataFormat: 'channelsFirst'|
    'channelsLast'): {padInfo: PadInfo, outHeight: number, outWidth: number} {
  let padInfo: PadInfo;
  let outHeight: number;
  let outWidth: number;

  if (typeof pad === 'number') {
    const padType = (pad === 0) ? 'VALID' : 'NUMBER';
    padInfo = {top: pad, bottom: pad, left: pad, right: pad, type: padType};
    const outShape = computeOutputShape2D(
        [inHeight, inWidth], filterHeight, strideHeight, pad, roundingMode);
    outHeight = outShape[0];
    outWidth = outShape[1];
  } else if (pad === 'same') {
    outHeight = Math.ceil(inHeight / strideHeight);
    outWidth = Math.ceil(inWidth / strideWidth);
    const padAlongHeight =
        Math.max(0, (outHeight - 1) * strideHeight + filterHeight - inHeight);
    const padAlongWidth =
        Math.max(0, (outWidth - 1) * strideWidth + filterWidth - inWidth);
    const top = Math.floor(padAlongHeight / 2);
    const bottom = padAlongHeight - top;
    const left = Math.floor(padAlongWidth / 2);
    const right = padAlongWidth - left;
    padInfo = {top, bottom, left, right, type: 'SAME'};
  } else if (pad === 'valid') {
    padInfo = {top: 0, bottom: 0, left: 0, right: 0, type: 'VALID'};
    outHeight = Math.ceil((inHeight - filterHeight + 1) / strideHeight);
    outWidth = Math.ceil((inWidth - filterWidth + 1) / strideWidth);
  } else if (typeof pad === 'object') {
    const top = dataFormat === 'channelsLast' ? pad[1][0] : pad[2][0];
    const bottom = dataFormat === 'channelsLast' ? pad[1][1] : pad[2][1];
    const left = dataFormat === 'channelsLast' ? pad[2][0] : pad[3][0];
    const right = dataFormat === 'channelsLast' ? pad[2][1] : pad[3][1];
    const padType = (top === 0 && bottom === 0 && left === 0 && right === 0) ?
        'VALID' :
        'EXPLICIT';
    padInfo = {top, bottom, left, right, type: padType};
    outHeight = conditionalRound(
        (inHeight - filterHeight + top + bottom) / strideHeight + 1,
        roundingMode);
    outWidth = conditionalRound(
        (inWidth - filterWidth + left + right) / strideWidth + 1, roundingMode);
  } else {
    throw Error(`Unknown padding parameter: ${pad}`);
  }
  return {padInfo, outHeight, outWidth};
}

function get3DPadAndOutInfo(
    pad: 'same'|'valid'|number, inDepth: number, inHeight: number,
    inWidth: number, strideDepth: number, strideHeight: number,
    strideWidth: number, filterDepth: number, filterHeight: number,
    filterWidth: number, roundingMode?: 'floor'|'round'|'ceil'): {
  padInfo: PadInfo3D,
  outDepth: number,
  outHeight: number,
  outWidth: number
} {
  let padInfo: PadInfo3D;
  let outDepth: number;
  let outHeight: number;
  let outWidth: number;

  if (typeof pad === 'number') {
    const padType = (pad === 0) ? 'VALID' : 'NUMBER';
    padInfo = {
      top: pad,
      bottom: pad,
      left: pad,
      right: pad,
      front: pad,
      back: pad,
      type: padType
    };
    const outShape = computeOutputShape4D(
        [inDepth, inHeight, inWidth, 1], filterDepth, 1, strideDepth, pad,
        roundingMode);
    outDepth = outShape[0];
    outHeight = outShape[1];
    outWidth = outShape[2];
  } else if (pad === 'same') {
    outDepth = Math.ceil(inDepth / strideDepth);
    outHeight = Math.ceil(inHeight / strideHeight);
    outWidth = Math.ceil(inWidth / strideWidth);
    const padAlongDepth = (outDepth - 1) * strideDepth + filterDepth - inDepth;
    const padAlongHeight =
        (outHeight - 1) * strideHeight + filterHeight - inHeight;
    const padAlongWidth = (outWidth - 1) * strideWidth + filterWidth - inWidth;
    const front = Math.floor(padAlongDepth / 2);
    const back = padAlongDepth - front;
    const top = Math.floor(padAlongHeight / 2);
    const bottom = padAlongHeight - top;
    const left = Math.floor(padAlongWidth / 2);
    const right = padAlongWidth - left;

    padInfo = {top, bottom, left, right, front, back, type: 'SAME'};
  } else if (pad === 'valid') {
    padInfo = {
      top: 0,
      bottom: 0,
      left: 0,
      right: 0,
      front: 0,
      back: 0,
      type: 'VALID'
    };
    outDepth = Math.ceil((inDepth - filterDepth + 1) / strideDepth);
    outHeight = Math.ceil((inHeight - filterHeight + 1) / strideHeight);
    outWidth = Math.ceil((inWidth - filterWidth + 1) / strideWidth);
  } else {
    throw Error(`Unknown padding parameter: ${pad}`);
  }
  return {padInfo, outDepth, outHeight, outWidth};
}

/**
 * Rounds a value depending on the rounding mode
 * @param value
 * @param roundingMode
 */
function conditionalRound(
    value: number, roundingMode?: 'floor'|'round'|'ceil') {
  if (!roundingMode) {
    return value;
  }
  switch (roundingMode) {
    case 'round':
      // used for Caffe Conv
      return Math.round(value);
    case 'ceil':
      // used for Caffe Pool
      return Math.ceil(value);
    case 'floor':
      return Math.floor(value);
    default:
      throw new Error(`Unknown roundingMode ${roundingMode}`);
  }
}

export function tupleValuesAreOne(param: number|number[]): boolean {
  const [dimA, dimB, dimC] = parseTupleParam(param);
  return dimA === 1 && dimB === 1 && dimC === 1;
}

export function eitherStridesOrDilationsAreOne(
    strides: number|number[], dilations: number|number[]): boolean {
  return tupleValuesAreOne(strides) || tupleValuesAreOne(dilations);
}

/**
 * Convert Conv2D dataFormat from 'NHWC'|'NCHW' to
 *    'channelsLast'|'channelsFirst'
 * @param dataFormat in 'NHWC'|'NCHW' mode
 * @return dataFormat in 'channelsLast'|'channelsFirst' mode
 * @throws unknown dataFormat
 */
export function convertConv2DDataFormat(dataFormat: 'NHWC'|'NCHW'):
    'channelsLast'|'channelsFirst' {
  if (dataFormat === 'NHWC') {
    return 'channelsLast';
  } else if (dataFormat === 'NCHW') {
    return 'channelsFirst';
  } else {
    throw new Error(`Unknown dataFormat ${dataFormat}`);
  }
}
