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
import * as concat_util from './concat_util';
import {operation} from './decorators';
import {Array1D, Array2D, Array3D, Array4D, NDArray} from './ndarray';
import {DataType, Rank, RankMap} from './types';

export class Ops {
  /**
   * Concatenates two 1D arrays.
   *
   * For example, if:
   * A: shape(3) = |r1, g1, b1|
   * B: shape(2) = |r2, g2|
   * C = concat1D(A, B) == |r1, g1, b1, r2, g2|
   *
   * @param a The first array.
   * @param b The second array.
   * @return The concatenated array.
   */
  @operation
  static concat1D<D extends DataType>(a: Array1D<D>, b: Array1D<D>):
      Array1D<D> {
    concat_util.assertParams(a.shape, b.shape, 0);
    return ENV.engine.executeKernel('Concat1D', {inputs: {a, b}}) as Array1D<D>;
  }

  /**
   * Concatenates two 2D arrays along a given axis.
   *
   * For example, if:
   * A: shape(2, 3) = | r1, g1, b1 |
   *                  | r2, g2, b2 |
   *
   * B: shape(2, 3) = | r3, g3, b3 |
   *                  | r4, g4, b4 |
   *
   * C = concat2D(A, B, axis)
   *
   * if axis = 0:
   * C: shape(4, 3) = | r1, g1, b1 |
   *                  | r2, g2, b2 |
   *                  | r3, g3, b3 |
   *                  | r4, g4, b4 |
   *
   * if axis = 1:
   * C = shape(2, 6) = | r1, g1, b1, r3, g3, b3 |
   *                   | r2, g2, b2, r4, g4, b4 |
   *
   *
   * @param a The first array.
   * @param b The second array.
   * @param axis The axis to concatenate along.
   * @return The concatenated array.
   */
  @operation
  static concat2D<D extends DataType>(
      a: Array2D<D>, b: Array2D<D>, axis: number): Array2D<D> {
    concat_util.assertParams(a.shape, b.shape, axis);
    return ENV.engine.executeKernel(
               'Concat2D', {inputs: {a, b}, args: {axis}}) as Array2D<D>;
  }

  /**
   * Concatenates two 3D ndarrays along a given axis.
   *
   * For example, if:
   * A: shape(2, 1, 3) = | r1, g1, b1 |
   *                     | r2, g2, b2 |
   *
   * B: shape(2, 1, 3) = | r3, g3, b3 |
   *                     | r4, g4, b4 |
   *
   * C = concat3D(A, B, axis)
   *
   * if axis = 0:
   * C: shape(4, 1, 3) = | r1, g1, b1 |
   *                     | r2, g2, b2 |
   *                     | r3, g3, b3 |
   *                     | r4, g4, b4 |
   *
   * if axis = 1:
   * C: shape(2, 2, 3) = | r1, g1, b1, r3, g3, b3 |
   *                     | r2, g2, b2, r4, g4, b4 |
   *
   * if axis = 2:
   * C = shape(2, 1, 6) = | r1, g1, b1, r3, g3, b3 |
   *                      | r2, g2, b2, r4, g4, b4 |
   *
   * @param a The first array to concat.
   * @param b The second array to conat.
   * @param axis The axis to concate along.
   * @return The concatenated array.
   */
  @operation
  static concat3D<D extends DataType>(
      a: Array3D<D>, b: Array3D<D>, axis: number): Array3D<D> {
    concat_util.assertParams(a.shape, b.shape, axis);

    const gradients = (dy: Array3D<'float32'>, y: Array3D) => {
      const {x1Begin, x1Size, x2Begin, x2Size} =
          concat_util.computeGradientSliceShapes3D(a.shape, y.shape, axis);
      return {
        a: () => dy.slice(x1Begin, x1Size),
        b: () => dy.slice(x2Begin, x2Size)
      };
    };

    return ENV.engine.executeKernel(
               'Concat3D', {inputs: {a, b}, args: {axis}}, gradients) as
        Array3D<D>;
  }

  /**
   * Concatenates two 4D ndarrays along a given axis. See math.concat2D() for
   * documentation.
   *
   * @param a The first array to concat.
   * @param b The second array to conat.
   * @param axis The axis to concate along.
   * @return The concatenated array.
   */
  @operation
  static concat4D<D extends DataType>(
      a: Array4D<D>, b: Array4D<D>, axis: number): Array4D<D> {
    concat_util.assertParams(a.shape, b.shape, axis);
    return ENV.engine.executeKernel(
               'Concat4D', {inputs: {a, b}, args: {axis}}) as Array4D<D>;
  }

  @operation
  static concat<D extends DataType, R extends Rank>(
      a: NDArray<D, R>, b: NDArray<D, R>, axis: number): RankMap<D>[R] {
    concat_util.assertParams(a.shape, b.shape, axis);
    if (a.rank === 0) {
      throw new Error('Cannot concatenate a scalar');
    } else if (a.rank === 1) {
      return a.concat(b, axis);
    } else if (a.rank === 2) {
      return a.concat(b, axis);
    } else if (a.rank === 3) {
      return a.concat(b, axis);
    } else if (a.rank === 4) {
      return a.concat(b, axis);
    } else {
      throw new Error(`Concat for rank ${a.rank} is not yet implemented`);
    }
  }
}
