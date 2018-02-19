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
import {Rank, ShapeMap} from '../types';
import * as util from '../util';
import {operation} from './operation';
import * as slice_util from './slice_util';

export class SliceOps {
  /**
   * Extracts a 1D slice from 1D array starting at coordinates `begin` and is
   * of length `size`. See `slice` for details.
   */
  static slice1d(x: Tensor1D, begin: number, size: number): Tensor1D {
    util.assert(
        x.rank === 1,
        `slice1d expects a rank-1 tensor, but got a rank-${x.rank} tensor`);
    return SliceOps.slice(x, [begin], [size]);
  }

  /**
   * Extracts a 2D slice from a 2D array starting at coordinates `begin` and
   * is of size `size`. See `slice` for details.
   */
  static slice2d(x: Tensor2D, begin: [number, number], size: [number, number]):
      Tensor2D {
    util.assert(
        x.rank === 2,
        `slice1d expects a rank-2 tensor, but got a rank-${x.rank} tensor`);
    return SliceOps.slice(x, begin, size);
  }

  /**
   * Extracts a 3D slice from a 3D array starting at coordinates `begin` and
   * is of size `size`. See `slice` for details.
   */
  static slice3d(x: Tensor3D, begin: [number, number, number], size: [
    number, number, number
  ]): Tensor3D {
    util.assert(
        x.rank === 3,
        `slice1d expects a rank-3 tensor, but got a rank-${x.rank} tensor`);
    return SliceOps.slice(x, begin, size);
  }

  /**
   * Extracts a 4D slice from a 4D array starting at coordinates `begin` and
   * is of size `size`. See `slice` for details.
   */
  static slice4d(x: Tensor4D, begin: [number, number, number, number], size: [
    number, number, number, number
  ]): Tensor4D {
    util.assert(
        x.rank === 4,
        `slice1d expects a rank-4 tensor, but got a rank-${x.rank} tensor`);
    return SliceOps.slice(x, begin, size);
  }

  /**
   * Extracts a slice from a `Tensor` starting at coordinates `begin`
   * and is of size `size`.
   *
   * Also available are stricter rank-specific methods with the same signature
   * as this method that assert that `x` is of the given rank:
   *   - `dl.slice1d`
   *   - `dl.slice2d`
   *   - `dl.slice3d`
   *   - `dl.slice4d`
   *
   * ```js
   * const x = dl.tensor1d([1, 2, 3, 4]);
   *
   * x.slice([1], [2]).print();
   * ```
   *
   * ```js
   * const x = dl.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * x.slice([1, 0], [1, 2]).print();
   * ```
   * @param x The input `Tensor` to slice from.
   * @param begin The coordinates to start the slice from. The length of this
   *     array should match the rank of `x`.
   * @param size The size of the slice. The length of this array should match
   *     the rank of `x`.
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static slice<R extends Rank, T extends Tensor<R>>(
      x: T, begin: ShapeMap[R], size: ShapeMap[R]): T {
    slice_util.assertParamsValid(x, begin, size);
    if (x.rank === 0) {
      throw new Error('Slicing scalar is not possible');
    }
    const inputShape = x.shape;
    const grad = (dy: T) => {
      // Create an Nx2 padding where the first column represents how many
      // zeros are prepended (at start) for each dimension, and the second
      // column indicates how many zeros are appended (at end).

      // The number of zeros to append is the shape of the input
      // elementwise-subtracted by both the begin vector and sizes vector.
      const paddings: Array<[number, number]> = [];
      for (let i = 0; i < dy.rank; i++) {
        paddings.push([begin[i], inputShape[i] - begin[i] - size[i]]);
      }
      return {x: () => dy.pad(paddings)};
    };
    return ENV.engine.runKernel(
               backend => backend.slice(x, begin, size), {x}, grad) as T;
  }
}
