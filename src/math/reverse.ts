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
import * as axis_util from './axis_util';
import {operation} from './decorators';
import {Array1D, Array2D, Array3D, Array4D} from './ndarray';

export class Ops {
  /**
   * Reverses a 1D array
   * @param x The input array.
   */
  @operation
  static reverse1D(x: Array1D): Array1D {
    util.assert(x.rank === 1, `Error in reverse1D: x must be rank 1 but got
             rank ${x.rank}.`);
    const input4D = x.as4D(1, 1, 1, x.shape[0]);
    const res = Ops.reverse4D(input4D, [3]);
    return res.as1D();
  }

  /**
   * Reverses a 2D array along a specified axis
   * @param x The input array.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)).
   */
  @operation
  static reverse2D(x: Array2D, axis: number|number[]): Array2D {
    util.assert(x.rank === 2, `Error in reverse2D: x must be rank 2 but got
             rank ${x.rank}.`);
    const axisCleaned = axis_util.parseAxisParam(axis, x.shape).map(a => a + 2);
    const input4D = x.as4D(1, 1, x.shape[0], x.shape[1]);
    const res = Ops.reverse4D(input4D, axisCleaned);
    return res.as2D(res.shape[2], res.shape[3]);
  }

  /**
   * Reverses a 3D array along a specified axis
   * @param x The input array.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)).
   */
  @operation
  static reverse3D(x: Array3D, axis: number|number[]): Array3D {
    util.assert(x.rank === 3, `Error in reverse3D: x must be rank 3 but got
             rank ${x.rank}.`);
    const axisCleaned = axis_util.parseAxisParam(axis, x.shape).map(a => a + 1);
    const input4D = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
    const res = Ops.reverse4D(input4D, axisCleaned);
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]);
  }

  /**
   * Reverses a 4D array along a specified axis
   * @param x The input array.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)).
   */
  @operation
  static reverse4D(x: Array4D, axis: number|number[]): Array4D {
    util.assert(x.rank === 4, `Error in reverse4D: x must be rank 4 but got
             rank ${x.rank}.`);
    const axisCleaned = axis_util.parseAxisParam(axis, x.shape);
    return ENV.engine.executeKernel(
        'Reverse4D', {inputs: {x}, args: {axis: axisCleaned}});
  }
}
