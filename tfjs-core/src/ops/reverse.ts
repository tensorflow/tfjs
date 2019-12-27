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
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';
import {op} from './operation';

/**
 * Reverses a `tf.Tensor1D`.
 *
 * @param x The input tensor.
 */
function reverse1d_(x: Tensor1D|TensorLike): Tensor1D {
  const $x = convertToTensor(x, 'x', 'reverse');
  util.assert(
      $x.rank === 1,
      () => `Error in reverse1D: x must be rank 1 but got rank ${$x.rank}.`);
  return reverse($x, 0);
}

/**
 * Reverses a `tf.Tensor2D` along a specified axis.
 *
 * @param x The input tensor.
 * @param axis The set of dimensions to reverse. Must be in the
 *     range [-rank(x), rank(x)). Defaults to all axes.
 */
function reverse2d_(x: Tensor2D|TensorLike, axis?: number|number[]): Tensor2D {
  const $x = convertToTensor(x, 'x', 'reverse');
  util.assert(
      $x.rank === 2,
      () => `Error in reverse2D: x must be rank 2 but got rank ${$x.rank}.`);
  return reverse($x, axis);
}

/**
 * Reverses a `tf.Tensor3D` along a specified axis.
 *
 * @param x The input tensor.
 * @param axis The set of dimensions to reverse. Must be in the
 *     range [-rank(x), rank(x)). Defaults to all axes.
 */
function reverse3d_(x: Tensor3D|TensorLike, axis?: number|number[]): Tensor3D {
  const $x = convertToTensor(x, 'x', 'reverse');
  util.assert(
      $x.rank === 3,
      () => `Error in reverse3D: x must be rank 3 but got rank ${$x.rank}.`);
  return reverse($x, axis);
}

/**
 * Reverses a `tf.Tensor4D` along a specified axis.
 *
 * @param x The input tensor.
 * @param axis The set of dimensions to reverse. Must be in the
 *     range [-rank(x), rank(x)). Defaults to all axes.
 */
function reverse4d_(x: Tensor4D|TensorLike, axis?: number|number[]): Tensor4D {
  const $x = convertToTensor(x, 'x', 'reverse');
  util.assert(
      $x.rank === 4,
      () => `Error in reverse4D: x must be rank 4 but got rank ${$x.rank}.`);
  return reverse($x, axis);
}

/**
 * Reverses a `tf.Tensor` along a specified axis.
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
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
function reverse_<T extends Tensor>(
    x: T|TensorLike, axis?: number|number[]): T {
  const $x = convertToTensor(x, 'x', 'reverse');

  if ($x.rank === 0) {
    return $x.clone();
  }
  const axes = util.parseAxisParam(axis, $x.shape);
  const grad = (dy: T) => {
    return {$x: () => dy.reverse(axes)};
  };
  const res =
      ENGINE.runKernelFunc(backend => backend.reverse($x, axes), {$x}, grad);
  return res.reshapeAs($x);
}

export const reverse = op({reverse_});
export const reverse1d = op({reverse1d_});
export const reverse2d = op({reverse2d_});
export const reverse3d = op({reverse3d_});
export const reverse4d = op({reverse4d_});
