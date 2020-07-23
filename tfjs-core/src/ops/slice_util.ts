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

import {Tensor} from '../tensor';
import * as util from '../util';

export function assertParamsValid(
    input: Tensor, begin: number[], size: number[]): void {
  util.assert(
      input.rank === begin.length,
      () => `Error in slice${input.rank}D: Length of begin ${begin} must ` +
          `match the rank of the array (${input.rank}).`);
  util.assert(
      input.rank === size.length,
      () => `Error in slice${input.rank}D: Length of size ${size} must ` +
          `match the rank of the array (${input.rank}).`);

  for (let i = 0; i < input.rank; ++i) {
    util.assert(
        begin[i] + size[i] <= input.shape[i],
        () => `Error in slice${input.rank}D: begin[${i}] + size[${i}] ` +
            `(${begin[i] + size[i]}) would overflow input.shape[${i}] (${
                  input.shape[i]})`);
  }
}

/** Converts a binary mask to an array of axes. Used in stridedSlice(). */
export function maskToAxes(mask: number): number[] {
  const axes = [];
  let axis = 0;
  while (mask > 0) {
    if (mask & 1) {
      axes.push(axis);
    }
    mask /= 2;
    axis++;
  }
  return axes;
}

/** Computes the output shape given the strided slice params. */
export function computeOutShape(
    begin: number[], end: number[], strides: number[]): number[] {
  const size = [];
  for (let axis = 0; axis < begin.length; axis++) {
    size[axis] = Math.ceil((end[axis] - begin[axis]) / strides[axis]);
  }
  return size;
}

// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current stride value. Otherwise, insert.
export function stridesWithElidedDims(
    strides: number[], ellipsisInsertionIndex: number, numElidedAxes: number,
    inputShape: number[]): number[] {
  const newStrides = [...strides];
  for (let i = newStrides.length; i < inputShape.length; i++) {
    newStrides.push(1);
  }
  for (let i = 0; i < numElidedAxes; i++) {
    if (i === 0) {
      newStrides[ellipsisInsertionIndex] = 1;
    } else {
      newStrides.splice(
          ellipsisInsertionIndex, 0 /* num elements to delete */,
          1 /* element to add */);
      newStrides.pop();
    }
  }
  return newStrides;
}

function unnormalizeAxis(
    ellipsisInsertionIndex: number, numElidedAxes: number,
    normalizedAxis: number): number {
  if (normalizedAxis <= ellipsisInsertionIndex) {
    return normalizedAxis;
  }

  return normalizedAxis - (numElidedAxes - 1);
}

function getElidedAxes(numElidedAxes: number, ellipsisInsertionIndex: number) {
  const elidedAxes = [];
  for (let i = 0; i < numElidedAxes; i++) {
    elidedAxes.push(ellipsisInsertionIndex + i);
  }
  return elidedAxes;
}

// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current start value. Otherwise, insert.
export function startIndicesWithElidedDims(
    beginMask: number, ellipsisInsertionIndex: number, numElidedAxes: number,
    originalBegin: number[], inputShape: number[]): number[] {
  const newIndices = [...inputShape];
  const elidedAxes = getElidedAxes(numElidedAxes, ellipsisInsertionIndex);

  for (let axis = 0; axis < newIndices.length; axis++) {
    if (elidedAxes.indexOf(axis) > -1) {
      newIndices[axis] = 0;
    } else {
      const originalAxis =
          unnormalizeAxis(ellipsisInsertionIndex, numElidedAxes, axis);
      let originalValue = originalBegin[originalAxis];
      if (beginMask & 1 << originalAxis) {
        originalValue = 0;
      }

      newIndices[axis] = originalValue;
    }
  }
  return newIndices;
}

// Creates full selection at the elided dimensions. If the dimension matches
// the ellipsis mask, override the current stop value. Otherwise, insert.
export function stopIndicesWithElidedDims(
    endMask: number, ellipsisInsertionIndex: number, numElidedAxes: number,
    originalEnd: number[], inputShape: number[]): number[] {
  const newIndices = [...inputShape];
  const elidedAxes = getElidedAxes(numElidedAxes, ellipsisInsertionIndex);

  for (let axis = 0; axis < newIndices.length; axis++) {
    if (elidedAxes.indexOf(axis) > -1) {
      newIndices[axis] = Number.MAX_SAFE_INTEGER;
    } else {
      const originalAxis =
          unnormalizeAxis(ellipsisInsertionIndex, numElidedAxes, axis);
      let originalValue = originalEnd[originalAxis];
      if (endMask & 1 << originalAxis) {
        originalValue = Number.MAX_SAFE_INTEGER;
      }
      newIndices[axis] = originalValue;
    }
  }

  for (let i = 0; i < newIndices.length; i++) {
    // Handle negative indices
    const axisSize = inputShape[i];
    if (newIndices[i] < 0) {
      newIndices[i] += axisSize;
    }
    newIndices[i] = util.clamp(0, newIndices[i], inputShape[i]);
  }
  return newIndices;
}

export function stridesForAxis(
    strides: number[], axis: number, ellipsisMask: number): number {
  let stride = strides[axis];
  if (ellipsisMask & (1 << axis) || stride == null) {
    stride = 1;
  }

  return stride;
}

export function startForAxis(
    beginMask: number, startIndices: number[], strides: number[],
    inputShape: number[], axis: number, ellipsisMask: number): number {
  // Begin with the specified index
  let start = startIndices[axis];
  const stride = strides[axis] || 1;

  // Check the axis bit from right of masked axes, or the begin index is not set
  // for the axis.
  if (beginMask & 1 << axis || ellipsisMask & 1 << axis || start == null) {
    if (stride > 0) {
      // Forward iteration - use the first element. These values will get
      // clamped below (Note: We could have set them to 0 and axis_size-1, but
      // use lowest() and max() to maintain symmetry with StopForAxis())
      start = Number.MIN_SAFE_INTEGER;
    } else {
      // Backward iteration - use the last element.
      start = Number.MAX_SAFE_INTEGER;
    }
  }

  // Handle negative indices
  const axisSize = inputShape[axis];
  if (start < 0) {
    start += axisSize;
  }

  // Clamping
  start = util.clamp(0, start, axisSize - 1);

  return start;
}

export function stopForAxis(
    endMask: number, stopIndices: number[], strides: number[],
    inputShape: number[], axis: number, ellipsisMask: number): number {
  // Begin with the specified index
  let stop = stopIndices[axis];
  const stride = strides[axis] || 1;

  // Check the axis bit from right of masked axes, or if the stop index is not
  // set for this axis.
  if (endMask & (1 << axis) || ellipsisMask & (1 << axis) || stop == null) {
    if (stride > 0) {
      // Forward iteration - use the last element. These values will get
      // clamped below
      stop = Number.MAX_SAFE_INTEGER;
    } else {
      // Backward iteration - use the first element.
      stop = Number.MIN_SAFE_INTEGER;
    }
  }

  // Handle negative indices
  const axisSize = inputShape[axis];
  if (stop < 0) {
    stop += axisSize;
  }

  // Clamping
  // Because the end index points one past the last element, we need slightly
  // different clamping ranges depending on the direction.
  if (stride > 0) {
    // Forward iteration
    stop = util.clamp(0, stop, axisSize);
  } else {
    // Backward iteration
    stop = util.clamp(-1, stop, axisSize - 1);
  }

  return stop;
}

/**
 * Returns true if the slice occupies a continous set of elements in the
 * 'flat' space.
 */
export function isSliceContinous(
    shape: number[], begin: number[], size: number[]) {
  // Index of the first axis that has size > 1.
  let firstNonOneAxis = size.length;
  for (let i = 0; i < size.length; i++) {
    if (size[i] > 1) {
      firstNonOneAxis = i;
      break;
    }
  }

  for (let i = firstNonOneAxis + 1; i < size.length; i++) {
    if (begin[i] > 0 || size[i] !== shape[i]) {
      return false;
    }
  }
  return true;
}

export function computeFlatOffset(begin: number[], strides: number[]): number {
  let flatOffset = begin.length > 0 ? begin[begin.length - 1] : 1;
  for (let i = 0; i < begin.length - 1; i++) {
    flatOffset += begin[i] * strides[i];
  }
  return flatOffset;
}

export function parseSliceParams(
    x: Tensor, begin: number|number[], size?: number|number[]) {
  // The following logic allows for more ergonomic calls.
  let begin_: number[];
  if (typeof begin === 'number') {
    begin_ = [begin, ...new Array(x.rank - 1).fill(0)];
  } else if (begin.length < x.rank) {
    begin_ = begin.concat(new Array(x.rank - begin.length).fill(0));
  } else {
    begin_ = begin.slice();
  }
  begin_.forEach(d => {
    util.assert(
        d !== -1, () => 'slice() does not support negative begin indexing.');
  });
  let size_: number[];
  if (size == null) {
    size_ = new Array(x.rank).fill(-1);
  } else if (typeof size === 'number') {
    size_ = [size, ...new Array(x.rank - 1).fill(-1)];
  } else if (size.length < x.rank) {
    size_ = size.concat(new Array(x.rank - size.length).fill(-1));
  } else {
    size_ = size;
  }
  size_ = size_.map((d, i) => {
    if (d >= 0) {
      return d;
    } else {
      util.assert(
          d === -1,
          () => `Negative size values should be exactly -1 but got ` +
              `${d} for the slice() size at index ${i}.`);
      return x.shape[i] - begin_[i];
    }
  });
  return [begin_, size_];
}
