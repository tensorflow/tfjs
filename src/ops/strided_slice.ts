/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {ENGINE} from '../engine';
import {Tensor} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import {op} from './operation';
import {slice} from './slice';
import {getStridedSlicedInfo} from './slice_util';

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
 * t = tf.tensor3d([1, 1, 1 ,2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
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
 */
/** @doc {heading: 'Operations', subheading: 'Slicing and Joining'} */
function stridedSlice_(
    x: Tensor|TensorLike, begin: number[], end: number[], strides: number[],
    beginMask = 0, endMask = 0, ellipsisMask = 0, newAxisMask = 0,
    shrinkAxisMask = 0): Tensor {
  if (ellipsisMask !== 0) {
    throw new Error('ellipsis mask is not yet supported');
  }
  if (newAxisMask !== 0) {
    throw new Error('new axis mask is not yet supported');
  }
  const $x = convertToTensor(x, 'x', 'stridedSlice');
  const nonStrided = strides.every(v => v === 1);
  if (nonStrided) {
    const [beginIndex, size, shrinkAxis] = getStridedSlicedInfo(
        $x.shape, begin, end, strides, beginMask, endMask, ellipsisMask,
        newAxisMask, shrinkAxisMask);
    const outShape =
        size.filter((_, index) => shrinkAxis.indexOf(index) === -1);
    return slice($x, beginIndex, size).reshape(outShape);
  }
  return ENGINE.runKernel(
      backend => backend.stridedSlice(
          $x, begin, end, strides, beginMask, endMask, ellipsisMask,
          newAxisMask, shrinkAxisMask),
      {$x});
}

export const stridedSlice = op({stridedSlice_});
