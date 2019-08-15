/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

import {ValueError} from '../errors';
import {PaddingMode} from '../keras_format/common';

import {pyListRepeat} from './generic_utils';
import {isInteger, max} from './math_utils';

/**
 * Transforms a single number of array of numbers into an array of numbers.
 * @param value
 * @param n: The size of the tuple to be returned.
 * @param name: Name of the parameter, used for generating error messages.
 * @returns An array of numbers.
 */
export function normalizeArray(
    value: number|number[], n: number, name: string): number[] {
  if (typeof value === 'number') {
    return pyListRepeat(value, n);
  } else {
    if (value.length !== n) {
      throw new ValueError(
          `The ${name} argument must be an integer or tuple of ${n} integers.` +
          ` Received: ${value.length} elements.`);
    }
    for (let i = 0; i < n; ++i) {
      const singleValue = value[i];
      if (!isInteger(singleValue)) {
        throw new ValueError(
            `The ${name} argument must be an integer or tuple of ${n}` +
            ` integers. Received: ${JSON.stringify(value)} including a` +
            ` non-integer number ${singleValue}`);
      }
    }
    return value;
  }
}

/**
 * Determines output length of a convolution given input length.
 * @param inputLength
 * @param filterSize
 * @param padding
 * @param stride
 * @param dilation: dilation rate.
 */
export function convOutputLength(
    inputLength: number, filterSize: number, padding: PaddingMode,
    stride: number, dilation = 1): number {
  if (inputLength == null) {
    return inputLength;
  }
  const dilatedFilterSize = filterSize + (filterSize - 1) * (dilation - 1);
  let outputLength: number;
  if (padding === 'same') {
    outputLength = inputLength;
  } else {  // VALID
    outputLength = inputLength - dilatedFilterSize + 1;
  }
  return Math.floor((outputLength + stride - 1) / stride);
}

export function deconvLength(
    dimSize: number, strideSize: number, kernelSize: number,
    padding: PaddingMode): number {
  if (dimSize == null) {
    return null;
  }

  if (padding === 'valid') {
    dimSize = dimSize * strideSize + max([kernelSize - strideSize, 0]);
  } else if (padding === 'same') {
    dimSize = dimSize * strideSize;
  } else {
    throw new ValueError(`Unsupport padding mode: ${padding}.`);
  }
  return dimSize;
}
