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

import {doc, operation} from './decorators';
import * as slice_util from './slice_util';
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from './tensor';
import {Rank, ShapeMap} from './types';

export class Ops {
  /**
   * Extracts a 1D slice from 1D array starting at coordinates `begin` and is
   * of length `size`.
   *
   * @param x The input array to slice from.
   * @param begin The offset to start the slice from.
   * @param size The size of the slice.
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static slice1d(x: Tensor1D, begin: number, size: number): Tensor1D {
    slice_util.assertParamsValid(x, [begin], [size]);
    return ENV.engine.executeKernel(
               'Slice1D', {inputs: {x}, args: {begin, size}}) as Tensor1D;
  }

  /**
   * Extracts a 2D slice from a 2D array starting at coordinates `begin` and
   * is of size `size`.
   *
   * @param x The input array to slice from.
   * @param begin The [row, col] 2d coordinates to start the slice from.
   * @param size The size of the slice.
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static slice2d(x: Tensor2D, begin: [number, number], size: [number, number]):
      Tensor2D {
    slice_util.assertParamsValid(x, begin, size);
    return ENV.engine.executeKernel(
               'Slice2D', {inputs: {x}, args: {begin, size}}) as Tensor2D;
  }

  /**
   * Extracts a 3D slice from a 3D array starting at coordinates `begin` and
   * is of size `size`.
   *
   * @param x The input array to slice from.
   * @param begin The [row, col, depth] 3d coordinates to start the slice from.
   * @param size The size of the slice.
   */
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static slice3d(x: Tensor3D, begin: [number, number, number], size: [
    number, number, number
  ]): Tensor3D {
    slice_util.assertParamsValid(x, begin, size);
    return ENV.engine.executeKernel(
               'Slice3D', {inputs: {x}, args: {begin, size}}) as Tensor3D;
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
  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static slice4d(x: Tensor4D, begin: [number, number, number, number], size: [
    number, number, number, number
  ]): Tensor4D {
    slice_util.assertParamsValid(x, begin, size);
    return ENV.engine.executeKernel(
               'Slice4D', {inputs: {x}, args: {begin, size}}) as Tensor4D;
  }

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
