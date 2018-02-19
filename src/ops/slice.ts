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
import {operation} from './operation';
import * as slice_util from './slice_util';

export class Ops {
  /**
   * Extracts a 1D slice from 1D array starting at coordinates `begin` and is
   * of length `size`.
   *
   * @param x The input array to slice from.
   * @param begin The offset to start the slice from.
   * @param size The size of the slice.
   */
  @operation
  static slice1d(x: Tensor1D, begin: number, size: number): Tensor1D {
    slice_util.assertParamsValid(x, [begin], [size]);
    return ENV.engine.runKernel(backend => backend.slice1D(x, begin, size));
  }

  /**
   * Extracts a 2D slice from a 2D array starting at coordinates `begin` and
   * is of size `size`.
   *
   * @param x The input array to slice from.
   * @param begin The [row, col] 2d coordinates to start the slice from.
   * @param size The size of the slice.
   */
  @operation
  static slice2d(x: Tensor2D, begin: [number, number], size: [number, number]):
      Tensor2D {
    slice_util.assertParamsValid(x, begin, size);
    return ENV.engine.runKernel(backend => backend.slice2D(x, begin, size));
  }

  /**
   * Extracts a 3D slice from a 3D array starting at coordinates `begin` and
   * is of size `size`.
   *
   * @param x The input array to slice from.
   * @param begin The [row, col, depth] 3d coordinates to start the slice from.
   * @param size The size of the slice.
   */
  @operation
  static slice3d(x: Tensor3D, begin: [number, number, number], size: [
    number, number, number
  ]): Tensor3D {
    slice_util.assertParamsValid(x, begin, size);
    return ENV.engine.runKernel(backend => backend.slice3D(x, begin, size));
  }

  /**
   * Extracts a 4D slice from a 4D array starting at coordinates `begin` and
   * is of size `size`.
   *
   * @param x The input array to slice from.
   * @param begin The [row, col, depth, depth2] 4d coordinates to start the
   *              slice from.
   * @param size The size of the slice.
   */
  @operation
  static slice4d(x: Tensor4D, begin: [number, number, number, number], size: [
    number, number, number, number
  ]): Tensor4D {
    slice_util.assertParamsValid(x, begin, size);
    return ENV.engine.runKernel(backend => backend.slice4D(x, begin, size));
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
  static slice<R extends Rank>(
      x: Tensor<R>, begin: ShapeMap[R], size: ShapeMap[R]): Tensor<R> {
    if (x.rank === 0) {
      throw new Error('Slicing scalar is not possible');
    } else if (x.rank === 1) {
      return Ops.slice1d(x as Tensor1D, begin[0], size[0]) as Tensor<R>;
    } else if (x.rank === 2) {
      return Ops.slice2d(
                 x as Tensor2D, begin as [number, number],
                 size as [number, number]) as Tensor<R>;
    } else if (x.rank === 3) {
      return Ops.slice3d(
                 x as Tensor3D, begin as [number, number, number],
                 size as [number, number, number]) as Tensor<R>;
    } else if (x.rank === 4) {
      return Ops.slice4d(
                 x as Tensor4D, begin as [number, number, number, number],
                 size as [number, number, number, number]) as Tensor<R>;
    } else {
      throw new Error(`Slicing for rank ${x.rank} not implemented yet`);
    }
  }
}
