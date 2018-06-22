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

import {doc} from '../doc';
import {ENV} from '../environment';
import {Tensor} from '../tensor';
import {assertArgumentsAreTensors} from '../tensor_util';
import {operation} from './operation';

export class StridedSliceOps {
  /**
   * Extracts a strided slice of a tensor.
   *
   * Roughly speaking, this op extracts a slice of size (end-begin)/stride from
   * the given input_ tensor. Starting at the location specified by begin the
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
   * @param beginMask: If the ith bit of begin_mask is set, begin[i] is ignored
   *      and the fullest possible range in that dimension is used instead.
   * @param endMask: If the ith bit of end_mask is set, end[i] is ignored
   *      and the fullest possible range in that dimension is used instead.
   */
  @doc({heading: 'Operations', subheading: 'Slicing and Joining'})
  @operation
  static stridedSlice<T extends Tensor>(
      x: T, begin: number[], end: number[], strides: number[], beginMask = 0,
      endMask = 0): T {
    assertArgumentsAreTensors({x}, 'stridedSlice');

    return ENV.engine.runKernel(
               backend => backend.stridedSlice(
                   x, begin, end, strides, beginMask, endMask),
               {x}) as T;
  }
}
