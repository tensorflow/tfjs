/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {ENGINE, ForwardFunc} from '../engine';
import {StridedSlice, StridedSliceAttrs, StridedSliceInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';

import {op} from './operation';
import {reshape} from './reshape';
import {slice} from './slice';
import {computeOutShape, getNormalizedAxes, maskToAxes} from './slice_util';

/**
 * Extracts a strided slice of a tensor.
 *
 * Roughly speaking, this op extracts a slice of size (end-begin)/stride from
 * the given input tensor (x). Starting at the location specified by begin the
 * slice continues by adding stride to the index until all dimensions are not
 * less than end. Note that a stride can be negative, which causes a reverse
 * slice.
 *
 * ```js
 * const t = tf.tensor3d([1, 1, 1 ,2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
 *    [3, 2, 3]);
 * t.stridedSlice([1, 0, 0], [2, 1, 3], [1, 1, 1]).print()  // [[[3, 3, 3]]]
 * t.stridedSlice([1, 0, 0], [2, 2, 3], [1, 1, 1]).print()  // [[[3, 3, 3],
 *                                                     // [4, 4, 4]]]
 * t.stridedSlice([1, -1, 0], [2, -3, 3], [1, -1, 1]).print() // [[[4, 4, 4],
 *                                                     // [3, 3, 3]]]
 * ```
 *
 * @param x The tensor to stride slice.
 * @param begin The coordinates to start the slice from.
 * @param end: The coordinates to end the slice at.
 * @param strides: The size of the slice.
 * @param beginMask: If the ith bit of beginMask is set, begin[i] is ignored
 *      and the fullest possible range in that dimension is used instead.
 * @param endMask: If the ith bit of endMask is set, end[i] is ignored
 *      and the fullest possible range in that dimension is used instead.
 * @param shrinkAxisMask: a bitmask where bit i implies that
 * the ith specification should shrink the dimensionality. begin and end must
 * imply a slice of size 1 in the dimension.
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
function stridedSlice_(
    x: Tensor|TensorLike, begin: number[], end: number[], strides?: number[],
    beginMask = 0, endMask = 0, ellipsisMask = 0, newAxisMask = 0,
    shrinkAxisMask = 0): Tensor {
  let $x = convertToTensor(x, 'x', 'stridedSlice');

  const forward: ForwardFunc<Tensor> = (backend) => {
    if (strides == null) {
      strides = new Array(begin.length);
    }

    const ellipsisAxes = maskToAxes(ellipsisMask);
    if (ellipsisAxes.length > 1) {
      throw new Error('Multiple ellipses in slice is not allowed.');
    }

    if (ellipsisMask !== 0 && newAxisMask !== 0) {
      throw new Error(
          'Using both ellipsisMask and newAxisMask is not yet supported.');
    }

    if (ellipsisMask !== 0 && shrinkAxisMask !== 0) {
      throw new Error(
          'Using both ellipsisMask and shrinkAxisMask is not yet supported.');
    }

    const numInterpolatedAxes = $x.rank - begin.length;

    // Expand the dims of x based on the newAxisMask.
    const expandAxes = maskToAxes(newAxisMask);
    const newShape = $x.shape.slice();
    expandAxes.forEach(axis => {
      begin[axis] = 0;
      end[axis] = 1;
      newShape.splice(axis, 0, 1);
    });
    $x = reshape($x, newShape);

    const {
      begin: normalizedBegin,
      end: normalizedEnd,
      strides: normalizedStrides
    } =
        getNormalizedAxes(
            $x.shape, ellipsisAxes, numInterpolatedAxes, begin, end, strides,
            beginMask, endMask, ellipsisMask);
    begin = normalizedBegin;
    end = normalizedEnd;
    strides = normalizedStrides;

    const shrinkAxes = maskToAxes(shrinkAxisMask);
    // Adjust the ends based on the shrink mask.
    shrinkAxes.forEach(axis => {
      end[axis] = begin[axis] + 1;
      strides[axis] = 1;
    });

    // Figure out the output shape.
    const size = computeOutShape(begin, end, strides);
    // Remove the axes based on shrinkMask.
    const outShape = size.filter((_, axis) => shrinkAxes.indexOf(axis) === -1);

    const nonStrided = strides.every(v => v === 1);
    if (nonStrided) {
      return reshape(slice($x, begin, size), outShape);
    }

    const res = backend.stridedSlice($x, begin, end, strides);
    return reshape(res, outShape);
  };

  const inputs: StridedSliceInputs = {x: $x};
  const attrs: StridedSliceAttrs = {
    begin,
    end,
    strides,
    beginMask,
    endMask,
    ellipsisMask,
    newAxisMask,
    shrinkAxisMask
  };

  return ENGINE.runKernelFunc(
      forward, inputs as {} as NamedTensorMap, null /* grad */, StridedSlice,
      attrs as {} as NamedAttrMap);
}

export const stridedSlice = op({stridedSlice_});
