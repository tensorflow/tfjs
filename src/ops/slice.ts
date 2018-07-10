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
import {convertToTensor} from '../tensor_util';
import {Rank, TensorLike} from '../types';
import * as util from '../util';
import {op} from './operation';
import * as slice_util from './slice_util';

class SliceOps {
  /**
   * Extracts a 1D slice from 1D array starting at coordinates `begin` and is
   * of length `size`. See `slice` for details.
   */
  static slice1d(x: Tensor1D|TensorLike, begin: number, size: number):
      Tensor1D {
    const $x = convertToTensor(x, 'x', 'slice1d');
    util.assert(
        $x.rank === 1,
        `slice1d expects a rank-1 tensor, but got a rank-${$x.rank} tensor`);
    return SliceOps.slice($x, [begin], [size]);
  }

  /**
   * Extracts a 2D slice from a 2D array starting at coordinates `begin` and
   * is of size `size`. See `slice` for details.
   */
  static slice2d(x: Tensor2D|TensorLike, begin: [number, number], size: [
    number, number
  ]): Tensor2D {
    const $x = convertToTensor(x, 'x', 'slice2d');
    util.assert(
        $x.rank === 2,
        `slice1d expects a rank-2 tensor, but got a rank-${$x.rank} tensor`);
    return SliceOps.slice($x, begin, size);
  }

  /**
   * Extracts a 3D slice from a 3D array starting at coordinates `begin` and
   * is of size `size`. See `slice` for details.
   */
  static slice3d(
      x: Tensor3D|TensorLike, begin: [number, number, number],
      size: [number, number, number]): Tensor3D {
    const $x = convertToTensor(x, 'x', 'slice3d');
    util.assert(
        $x.rank === 3,
        `slice1d expects a rank-3 tensor, but got a rank-${$x.rank} tensor`);
    return SliceOps.slice($x, begin, size);
  }

  /**
   * Extracts a 4D slice from a 4D array starting at coordinates `begin` and
   * is of size `size`. See `slice` for details.
   */
  static slice4d(
      x: Tensor4D|TensorLike, begin: [number, number, number, number],
      size: [number, number, number, number]): Tensor4D {
    const $x = convertToTensor(x, 'x', 'slice4d');
    util.assert(
        $x.rank === 4,
        `slice1d expects a rank-4 tensor, but got a rank-${$x.rank} tensor`);
    return SliceOps.slice($x, begin, size);
  }

  /**
   * Extracts a slice from a `Tensor` starting at coordinates `begin`
   * and is of size `size`.
   *
   * Also available are stricter rank-specific methods with the same signature
   * as this method that assert that `x` is of the given rank:
   *   - `tf.slice1d`
   *   - `tf.slice2d`
   *   - `tf.slice3d`
   *   - `tf.slice4d`
   *
   * ```js
   * const x = tf.tensor1d([1, 2, 3, 4]);
   *
   * x.slice([1], [2]).print();
   * ```
   *
   * ```js
   * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * x.slice([1, 0], [1, 2]).print();
   * ```
   * @param x The input `Tensor` to slice from.
   * @param begin The coordinates to start the slice from. The length can be
   *     less than the rank of x - the rest of the axes will have implicit 0 as
   *     start. Can also be a single number, in which case it specifies the
   *     first axis.
   * @param size The size of the slice. The length can be less than the rank of
   *     x - the rest of the axes will have implicit -1. A value of -1 requests
   *     the rest of the dimensions in the axis. Can also be a single number,
   *     in which case it specifies the size of the first axis.
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  static slice<R extends Rank, T extends Tensor<R>>(
      x: T|TensorLike, begin: number|number[], size?: number|number[]): T {
    const $x = convertToTensor(x, 'x', 'slice');

    if ($x.rank === 0) {
      throw new Error('Slicing scalar is not possible');
    }
    // The following logic allows for more ergonomic calls.
    let begin_: number[];
    if (typeof begin === 'number') {
      begin_ = [begin, ...new Array($x.rank - 1).fill(0)];
    } else if (begin.length < $x.rank) {
      begin_ = begin.concat(new Array($x.rank - begin.length).fill(0));
    } else {
      begin_ = begin;
    }
    let size_: number[];
    if (size == null) {
      size_ = new Array($x.rank).fill(-1);
    } else if (typeof size === 'number') {
      size_ = [size, ...new Array($x.rank - 1).fill(-1)];
    } else if (size.length < $x.rank) {
      size_ = size.concat(new Array($x.rank - size.length).fill(-1));
    } else {
      size_ = size;
    }
    size_ = size_.map((d, i) => {
      if (d >= 0) {
        return d;
      } else {
        util.assert(d === -1, 'Bad value in size');
        return $x.shape[i] - begin_[i];
      }
    });
    slice_util.assertParamsValid($x, begin_, size_);
    const inputShape = $x.shape;
    const grad = (dy: T) => {
      // Create an Nx2 padding where the first column represents how many
      // zeros are prepended (at start) for each dimension, and the second
      // column indicates how many zeros are appended (at end).

      // The number of zeros to append is the shape of the input
      // elementwise-subtracted by both the begin vector and sizes vector.
      const paddings: Array<[number, number]> = [];
      for (let i = 0; i < dy.rank; i++) {
        paddings.push([begin_[i], inputShape[i] - begin_[i] - size_[i]]);
      }
      return {$x: () => dy.pad(paddings)};
    };
    return ENV.engine.runKernel(
               backend => backend.slice($x, begin_, size_), {$x}, grad) as T;
  }
}

export const slice = op(SliceOps.slice);
export const slice1d = op(SliceOps.slice1d);
export const slice2d = op(SliceOps.slice2d);
export const slice3d = op(SliceOps.slice3d);
export const slice4d = op(SliceOps.slice4d);
