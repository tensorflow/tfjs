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

import {MatrixOrientation} from './backends/types/matmul';
import {doc, operation} from './decorators';
import {Scalar, Tensor1D, Tensor2D} from './tensor';

export class Ops {
  /**
   * Computes the dot product of two matrices, A * B. These must be matrices,
   * use matrixTimesVector and vectorTimesMatrix, dotProduct, and outerProduct
   * in other cases.
   * @param a First matrix in dot product operation.
   * @param b Second matrix in dot product operation.
   * @param aOrientation The MatrixOrientation of A. If using TRANSPOSED, will
   * compute A^T * B.
   * @param bOrientation The MatrixOrientation of B. If using TRANSPOSED, will
   * compute A * B^T.
   */
  @doc({heading: 'Operations', subheading: 'Matrices'})
  @operation
  static matMul(
      a: Tensor2D, b: Tensor2D, aOrientation = MatrixOrientation.REGULAR,
      bOrientation = MatrixOrientation.REGULAR): Tensor2D {
    const innerShapeA =
        (aOrientation === MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];
    const innerShapeB =
        (bOrientation === MatrixOrientation.REGULAR) ? b.shape[0] : b.shape[1];

    util.assert(
        a.rank === 2 && b.rank === 2,
        `Error in matMul: inputs must be rank 2, got ranks ${a.rank}` +
            ` and ${b.rank}.`);

    util.assert(
        innerShapeA === innerShapeB,
        `Error in matMul: inner shapes (${innerShapeA}) and (` +
            `${innerShapeB}) of Tensors with shapes ${a.shape} and ` +
            `${b.shape} and orientations ${MatrixOrientation[aOrientation]}` +
            ` and ${MatrixOrientation[bOrientation]} must match.`);

    return ENV.engine.executeKernel(
               'MatMul', {inputs: {a, b}, args: {aOrientation, bOrientation}},
               (dy: Tensor2D, y: Tensor2D) => {
                 if (aOrientation === MatrixOrientation.TRANSPOSED ||
                     bOrientation === MatrixOrientation.TRANSPOSED) {
                   throw new Error(
                       `Backprop for transposed MatMul not yet implemented.`);
                 }
                 return {
                   a: () => dy.matMul(
                                b.toFloat(), MatrixOrientation.REGULAR,
                                MatrixOrientation.TRANSPOSED) as Tensor2D,
                   b: () => a.toFloat().matMul(
                                dy, MatrixOrientation.TRANSPOSED,
                                MatrixOrientation.REGULAR) as Tensor2D
                 };
               }) as Tensor2D;
  }

  /**
   * Computes the dot product of a vector and a matrix, v * B.
   * @param v The vector in dot product operation.
   * @param matrix The matrix in dot product operation.
   */
  @doc({heading: 'Operations', subheading: 'Matrices'})
  @operation
  static vectorTimesMatrix(v: Tensor1D, matrix: Tensor2D): Tensor1D {
    util.assert(
        v.rank === 1,
        `Error in vectorTimesMatrix: first input must be rank 1, but got ` +
            `rank ${v.rank}.`);
    util.assert(
        matrix.rank === 2,
        `Error in vectorTimesMatrix: second input must be rank 2, but got ` +
            `rank ${matrix.rank}.`);
    util.assert(
        v.size === matrix.shape[0],
        `Error in vectorTimesMatrix: size of vector (${v.size}) ` +
            `must match first dimension of matrix (${matrix.shape[0]})`);
    return v.as2D(1, -1).matMul(matrix).as1D();
  }

  /**
   * Computes the dot product of a matrix and vector, A * v.
   * @param matrix The matrix in dot product operation.
   * @param v The vector in dot product operation.
   */
  @doc({heading: 'Operations', subheading: 'Matrices'})
  @operation
  static matrixTimesVector(matrix: Tensor2D, v: Tensor1D): Tensor1D {
    util.assert(
        v.rank === 1,
        `Error in matrixTimesVector: second input must rank 1, but got ` +
            `rank ${v.rank}.`);
    util.assert(
        matrix.rank === 2,
        `Error in matrixTimesVector: first input must be a rank 2, but got ` +
            `rank ${matrix.rank}.`);
    util.assert(
        v.size === matrix.shape[1],
        `Error in matrixTimesVector: size of first rank 1 input ${v.size} ` +
            `must match inner dimension of second rank 2 input, but got ` +
            `shape ${matrix.shape}.`);

    return matrix.matMul(v.as2D(-1, 1)).as1D();
  }

  /**
   * Computes the dot product of two vectors, v1 * v2.
   * @param v1 The first vector in the dot product operation.
   * @param v2 The second vector in the dot product operation.
   */
  @doc({heading: 'Operations', subheading: 'Matrices'})
  @operation
  static dotProduct(v1: Tensor1D, v2: Tensor1D): Scalar {
    util.assert(
        v1.rank === 1 && v2.rank === 1,
        `Error in dotProduct: inputs must be rank 1, but got ranks ` +
            `${v1.rank} and ${v2.rank}.`);
    util.assert(
        v1.size === v2.size,
        `Error in dotProduct: size of inputs (${v1.size}) and (` +
            `${v2.size}) must match.`);
    return v1.as2D(1, -1).matMul(v2.as2D(-1, 1)).asScalar();
  }

  /**
   * Computes the outer product of two vectors, v1 and v2.
   * @param v1 The first vector in the outer product operation.
   * @param v2 The second vector in the dot product operation.
   */
  @doc({heading: 'Operations', subheading: 'Matrices'})
  @operation
  static outerProduct(v1: Tensor1D, v2: Tensor1D): Tensor2D {
    util.assert(
        v1.rank === 1 && v2.rank === 1,
        `Error in outerProduct: inputs must be rank 1, but got ranks ` +
            `${v1.rank} and ${v2.rank}.`);

    return v1.as2D(-1, 1).matMul(v2.as2D(1, -1));
  }
}
