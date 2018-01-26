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

import {ENV} from '../environment';
import * as util from '../util';
import * as broadcast_util from './broadcast_util';
import {operation} from './decorators';
import {NDArray} from './ndarray';
import * as types from './types';
import {DataType} from './types';

export class Ops {
  /**
   * Returns the truth value of a AND b element-wise. Supports broadcasting.
   *
   * @param a The first input `NDArray`.
   * @param b The second input `NDArray`.
   */
  @operation
  static logicalAnd(a: NDArray, b: NDArray): NDArray {
    util.assert(
        a.dtype === 'bool' && b.dtype === 'bool',
        'Error Array must be of type bool.');
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('LogicalAnd', {inputs: {a, b}});
  }

  /**
   * Returns the truth value of a OR b element-wise. Supports broadcasting.
   *
   * @param a The first input `NDArray`.
   * @param b The second input `NDArray`.
   */
  @operation
  static logicalOr(a: NDArray, b: NDArray): NDArray {
    util.assert(
        a.dtype === 'bool' && b.dtype === 'bool',
        'Error Array must be of type bool.');
    broadcast_util.assertAndGetBroadcastShape(a.shape, b.shape);
    return ENV.engine.executeKernel('LogicalOr', {inputs: {a, b}});
  }

  /**
   * Returns the elements, either `a` or `b` depending on the `condition`.
   *
   * @param condition The input as `NDAray<'bool'>.
   * @param a Input as `NDArray` which may have the same shape as
   *     `condition`. If `condition` is rank 1, `a` may have a higher rank but
   *     its first dimension must match the size of `condition`.
   * @param b Input as `NDArray` with the same shape and type as `a`.
   * @return An `NDArray` with the same type and shape as `a` and `b`.
   */
  @operation
  static where<T extends NDArray>(condition: NDArray, a: T, b: T): T {
    util.assert(
        condition.dtype === 'bool' || a.dtype === 'bool' || b.dtype === 'bool',
        'Error Array must be of type bool.');

    util.assertShapesMatch(a.shape, b.shape, 'Error in where: ');

    if (condition.rank === 1) {
      // If condition rank is 1, then the first dimension must match the size of
      // condition.
      util.assert(
          condition.shape[0] === a.shape[0],
          'The first dimension of `a` must match the size of `condition`.');
    } else {
      // A must have the same shape as condition.
      util.assertShapesMatch(condition.shape, b.shape, 'Error in where: ');
    }

    // Default to highest percision of number:
    const dtype = types.upcastType(a.dtype, b.dtype);
    return ENV.engine.executeKernel(
               'Where',
               {inputs: {condition, a, b}, args: {dtype: dtype as DataType}}) as
        T;
  }
}
