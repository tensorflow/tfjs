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
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import * as util from '../util';
import * as axis_util from './axis_util';
import {operation} from './operation';

export class Ops {
  /**
   * Reverses a `Tensor1D`.
   *
   * @param x The input tensor.
   */
  static reverse1d(x: Tensor1D): Tensor1D {
    util.assert(x.rank === 1, `Error in reverse1D: x must be rank 1 but got
             rank ${x.rank}.`);
    return Ops.reverse(x, 0);
  }

  /**
   * Reverses a `Tensor2D` along a specified axis
   *
   * @param x The input tensor.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)).
   */
  static reverse2d(x: Tensor2D, axis: number|number[]): Tensor2D {
    util.assert(x.rank === 2, `Error in reverse2D: x must be rank 2 but got
             rank ${x.rank}.`);
    return Ops.reverse(x, axis);
  }

  /**
   * Reverses a `Tensor3D` along a specified axis
   * @param x The input tensor.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)).
   */
  static reverse3d(x: Tensor3D, axis: number|number[]): Tensor3D {
    util.assert(x.rank === 3, `Error in reverse3D: x must be rank 3 but got
             rank ${x.rank}.`);
    return Ops.reverse(x, axis);
  }

  /**
   * Reverses a `Tensor4D` along a specified axis
   * @param x The input tensor.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)).
   */
  static reverse4d(x: Tensor4D, axis: number|number[]): Tensor4D {
    util.assert(x.rank === 4, `Error in reverse4D: x must be rank 4 but got
             rank ${x.rank}.`);
    return Ops.reverse(x, axis);
  }

  /**
   * Reverses a `Tensor` along a specified axis.
   *
   * @param x The input tensor.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)).
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static reverse<T extends Tensor>(x: T, axis: number|number[]): T {
    let x4d: Tensor4D;
    const axisCleaned =
        axis_util.parseAxisParam(axis, x.shape).map(a => a + 4 - x.rank);
    if (x.rank === 0) {
      return x.clone();
    } else if (x.rank === 1) {
      x4d = x.as4D(1, 1, 1, x.shape[0]);
    } else if (x.rank === 2) {
      x4d = x.as4D(1, 1, x.shape[0], x.shape[1]);
    } else if (x.rank === 3) {
      x4d = x.as4D(1, x.shape[0], x.shape[1], x.shape[2]);
    } else if (x.rank === 4) {
      x4d = x as Tensor4D;
    } else {
      throw new Error(`Reverse for rank ${x.rank} is not yet implemented`);
    }
    const res = ENV.engine.executeKernel(
        'Reverse4D', {inputs: {x: x4d}, args: {axis: axisCleaned}});
    return res.reshapeAs(x);
  }
}
