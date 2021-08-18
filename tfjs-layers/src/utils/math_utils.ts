/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/**
 * Math utility functions.
 *
 * This file contains some frequently used math function that operates on
 * number[] or Float32Array and return a number. Many of these functions are
 * not-so-thick wrappers around TF.js Core functions. But they offer the
 * convenience of
 * 1) not having to convert the inputs into Tensors,
 * 2) not having to convert the returned Tensors to numbers.
 */

import {ValueError} from '../errors';

export type ArrayTypes = Uint8Array|Int32Array|Float32Array;

/**
 * Determine if a number is an integer.
 */
export function isInteger(x: number): boolean {
  return x === parseInt(x.toString(), 10);
}

/**
 * Calculate the product of an array of numbers.
 * @param array The array to calculate the product over.
 * @param begin Beginning index, inclusive.
 * @param end Ending index, exclusive.
 * @return The product.
 */
export function arrayProd(
    array: number[]|ArrayTypes, begin?: number, end?: number): number {
  if (begin == null) {
    begin = 0;
  }
  if (end == null) {
    end = array.length;
  }

  let prod = 1;
  for (let i = begin; i < end; ++i) {
    prod *= array[i];
  }
  return prod;
}

/**
 * Compute minimum value.
 * @param array
 * @return minimum value.
 */
export function min(array: number[]|Float32Array): number {
  // same behavior as tf.min()
  if (array.length === 0) {
    return Number.NaN;
  }
  let min = Number.POSITIVE_INFINITY;
  for (let i = 0; i < array.length; i++) {
    const value = array[i];
    if (value < min) {
      min = value;
    }
  }
  return min;
}

/**
 * Compute maximum value.
 * @param array
 * @return maximum value
 */
export function max(array: number[]|Float32Array): number {
  // same behavior as tf.max()
  if (array.length === 0) {
    return Number.NaN;
  }
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < array.length; i++) {
    const value = array[i];
    if (value > max) {
      max = value;
    }
  }
  return max;
}

/**
 * Compute sum of array.
 * @param array
 * @return The sum.
 */
export function sum(array: number[]|Float32Array): number {
  let sum = 0;
  for (let i = 0; i < array.length; i++) {
    const value = array[i];
    sum += value;
  }
  return sum;
}

/**
 * Compute mean of array.
 * @param array
 * @return The mean.
 */
export function mean(array: number[]|Float32Array): number {
  return sum(array) / array.length;
}

/**
 * Compute variance of array.
 * @param array
 * @return The variance.
 */
export function variance(array: number[]|Float32Array): number {
  const meanValue = mean(array);
  const demeaned = array.map((value: number) => value - meanValue);
  let sumSquare = 0;
  for (let i = 0; i < demeaned.length; i++) {
    const value = demeaned[i];
    sumSquare += value * value;
  }
  return sumSquare / array.length;
}

/**
 * Compute median of array.
 * @param array
 * @return The median value.
 */
export function median(array: number[]|Float32Array): number {
  const arraySorted = array.slice().sort((a, b) => a - b);
  const lowIdx = Math.floor((arraySorted.length - 1) / 2);
  const highIdx = Math.ceil((arraySorted.length - 1) / 2);
  if (lowIdx === highIdx) {
    return arraySorted[lowIdx];
  }
  return (arraySorted[lowIdx] + arraySorted[highIdx]) / 2;
}

/**
 * Generate an array of integers in [begin, end).
 * @param begin Beginning integer, inclusive.
 * @param end Ending integer, exclusive.
 * @returns Range array.
 * @throws ValueError, iff `end` < `begin`.
 */
export function range(begin: number, end: number): number[] {
  if (end < begin) {
    throw new ValueError(`end (${end}) < begin (${begin}) is forbidden.`);
  }
  const out: number[] = [];
  for (let i = begin; i < end; ++i) {
    out.push(i);
  }
  return out;
}
