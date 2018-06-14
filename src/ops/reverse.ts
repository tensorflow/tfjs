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
import {parseAxisParam} from './axis_util';
import {operation} from './operation';

export class ReverseOps {
  /**
   * Reverses a `Tensor1D`.
   *
   * @param x The input tensor.
   */
  static reverse1d(x: Tensor1D): Tensor1D {
    util.assert(x.rank === 1, `Error in reverse1D: x must be rank 1 but got
             rank ${x.rank}.`);
    return ReverseOps.reverse(x, 0);
  }

  /**
   * Reverses a `Tensor2D` along a specified axis
   *
   * @param x The input tensor.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)). Defaults to all axes.
   */
  static reverse2d(x: Tensor2D, axis?: number|number[]): Tensor2D {
    util.assert(x.rank === 2, `Error in reverse2D: x must be rank 2 but got
             rank ${x.rank}.`);
    return ReverseOps.reverse(x, axis);
  }

  /**
   * Reverses a `Tensor3D` along a specified axis
   *
   * @param x The input tensor.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)). Defaults to all axes.
   */
  static reverse3d(x: Tensor3D, axis?: number|number[]): Tensor3D {
    util.assert(x.rank === 3, `Error in reverse3D: x must be rank 3 but got
             rank ${x.rank}.`);
    return ReverseOps.reverse(x, axis);
  }

  /**
   * Reverses a `Tensor4D` along a specified axis
   *
   * @param x The input tensor.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)). Defaults to all axes.
   */
  static reverse4d(x: Tensor4D, axis?: number|number[]): Tensor4D {
    util.assert(x.rank === 4, `Error in reverse4D: x must be rank 4 but got
             rank ${x.rank}.`);
    return ReverseOps.reverse(x, axis);
  }

  /**
   * Reverses a `Tensor` along a specified axis.
   *
   * Also available are stricter rank-specific methods that assert that `x` is
   * of the given rank:
   *   - `tf.reverse1d`
   *   - `tf.reverse2d`
   *   - `tf.reverse3d`
   *   - `tf.reverse4d`
   *
   * Except `tf.reverse1d` (which does not have axis param), all methods have
   * same signature as this method.
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   *
   * x.reverse().print();
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * const axis = 1;
   * x.reverse(axis).print();
   * ```
   * @param x The input tensor to be reversed.
   * @param axis The set of dimensions to reverse. Must be in the
   *     range [-rank(x), rank(x)). Defaults to all axes.
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static reverse<T extends Tensor>(x: T, axis?: number|number[]): T {
    util.assertArgumentsAreTensors({x}, 'reverse');

    if (x.rank === 0) {
      return x.clone();
    }
    const axes = parseAxisParam(axis, x.shape);
    const grad = (dy: T) => {
      return {x: () => dy.reverse(axes)};
    };
    const res =
        ENV.engine.runKernel(backend => backend.reverse(x, axes), {x}, grad);
    return res.reshapeAs(x);
  }
}
