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

import {operation} from './operation';
import {doc} from '../doc';
import {ENV} from '../environment';
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import * as util from '../util';
import * as concat_util from './concat_util';

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
  static concat1d(a: Tensor1D, b: Tensor1D): Tensor1D {
    return Ops.concat(a, b, 0 /* axis */);
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
  static concat2d(a: Tensor2D, b: Tensor2D, axis: number): Tensor2D {
    return Ops.concat(a, b, axis);
  }

  /**
   * Concatenates two 3D tensors along a given axis.
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
  static concat3d(a: Tensor3D, b: Tensor3D, axis: number): Tensor3D {
    return Ops.concat(a, b, axis);
  }

  /**
   * Concatenates two 4D tensors along a given axis. See dl.concat2D() for
   * documentation.
   *
   * @param a The first array to concat.
   * @param b The second array to conat.
   * @param axis The axis to concate along.
   * @return The concatenated array.
   */
  static concat4d(a: Tensor4D, b: Tensor4D, axis: number): Tensor4D {
    return Ops.concat(a, b, axis);
  }

  @doc({heading: 'Tensors', subheading: 'Slicing and Joining'})
  @operation
  static concat<T extends Tensor>(a: T, b: T, axis: number): T {
    concat_util.assertParams(a.shape, b.shape, axis);
    const outShape = concat_util.computeOutShape(a.shape, b.shape, axis);

    // Do the reshape.
    const a2D = a.as2D(-1, util.sizeFromShape(a.shape.slice(axis)));
    const b2D = b.as2D(-1, util.sizeFromShape(b.shape.slice(axis)));
    // Concats 2d tensors along axis=1. See comments in MathBackend.concat().
    const {aBegin, aSize, bBegin, bSize} =
        concat_util.computeGradientSliceShapes(a2D.shape, b2D.shape);
    const der = (dy: Tensor2D) => {
      return {
        a: () => dy.slice(aBegin, aSize),
        b: () => dy.slice(bBegin, bSize)
      };
    };
    const res =
        ENV.engine.executeKernel('Concat', {inputs: {a: a2D, b: b2D}}, der);
    return res.reshape(outShape) as T;
  }
}
