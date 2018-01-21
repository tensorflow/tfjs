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
import {Array2D} from './ndarray';

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
export function matMul(
    a: Array2D, b: Array2D, aOrientation = MatrixOrientation.REGULAR,
    bOrientation = MatrixOrientation.REGULAR): Array2D {
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
          `${innerShapeB}) of NDArrays with shapes ${a.shape} and ` +
          `${b.shape} and orientations ${MatrixOrientation[aOrientation]}` +
          ` and ${MatrixOrientation[bOrientation]} must match.`);

  return ENV.engine.executeKernel(
      'MatMul', {inputs: {a, b}, args: {aOrientation, bOrientation}},
      (dy: Array2D<'float32'>, y: Array2D) => {
        if (aOrientation === MatrixOrientation.TRANSPOSED ||
            bOrientation === MatrixOrientation.TRANSPOSED) {
          throw new Error(
              `Backprop for transposed MatMul not yet implemented.`);
        }
        return {
          a: () => matMul(
                       dy, b, MatrixOrientation.REGULAR,
                       MatrixOrientation.TRANSPOSED) as Array2D<'float32'>,
          b: () => matMul(
                       a, dy, MatrixOrientation.TRANSPOSED,
                       MatrixOrientation.REGULAR) as Array2D<'float32'>
        };
      });
}
