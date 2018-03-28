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
import {Scalar, Tensor1D, Tensor2D} from '../tensor';
import * as util from '../util';
import {operation} from './operation';

export class MatmulOps {
  /**
   * Computes the dot product of two matrices, A * B. These must be matrices.
   *
   * ```js
   * const a = tf.tensor2d([1, 2], [1, 2]);
   * const b = tf.tensor2d([1, 2, 3, 4], [2, 2]);
   *
   * a.matMul(b).print();  // or tf.matMul(a, b)
   * ```
   * @param a First matrix in dot product operation.
   * @param b Second matrix in dot product operation.
   * @param transposeA If true, `a` is transposed before multiplication.
   * @param transposeB If true, `b` is transposed before multiplication.
   */
  @doc({heading: 'Operations', subheading: 'Matrices'})
  @operation
  static matMul(
      a: Tensor2D, b: Tensor2D, transposeA = false, transposeB = false):
      Tensor2D {
    const innerShapeA = transposeA ? a.shape[0] : a.shape[1];
    const innerShapeB = transposeB ? b.shape[1] : b.shape[0];

    util.assert(
        a.rank === 2 && b.rank === 2,
        `Error in matMul: inputs must be rank 2, got ranks ${a.rank}` +
            ` and ${b.rank}.`);

    util.assert(
        innerShapeA === innerShapeB,
        `Error in matMul: inner shapes (${innerShapeA}) and (` +
            `${innerShapeB}) of Tensors with shapes ${a.shape} and ` +
            `${b.shape} and transposeA=${transposeA}` +
            ` and transposeB=${transposeB} must match.`);

    const grad = (dy: Tensor2D) => {
      if (transposeA || transposeB) {
        throw new Error(`Backprop for transposed MatMul not yet implemented.`);
      }
      return {
        a: () => dy.matMul(b.toFloat(), false, true),
        b: () => a.toFloat().matMul(dy, true, false)
      };
    };
    return ENV.engine.runKernel(
        backend => backend.matMul(a, b, transposeA, transposeB), {a, b}, grad);
  }

  /**
   * Computes the dot product of a vector and a matrix, v * B.
   *
   * @param v The vector in dot product operation.
   * @param matrix The matrix in dot product operation.
   */
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
   *
   * @param matrix The matrix in dot product operation.
   * @param v The vector in dot product operation.
   */
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
   *
   * @param v1 The first vector in the dot product operation.
   * @param v2 The second vector in the dot product operation.
   */
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
   *
   * ```js
   * const a = tf.tensor1d([1, 2, 3]);
   * const b = tf.tensor1d([3, 4, 5]);
   *
   * tf.outerProduct(a, b).print();
   * ```
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
