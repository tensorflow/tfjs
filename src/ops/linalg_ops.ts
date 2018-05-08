/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

/**
 * Linear algebra ops.
 */

import {doc} from '../doc';
import {Tensor1D, Tensor2D} from '../tensor';
import {Tracking} from '../tracking';
import {assert} from '../util';

import {operation} from './operation';
import {norm, split, squeeze, stack, sum} from './ops';

export class LinalgOps {
  /**
   * Gram-Schmidt orthogonalization.
   *
   * @param xs The vectors to be orthogonalized, in one of the two following
   *   formats:
   *   - An Array of `Tensor1D`.
   *   - A `Tensor2D`, i.e., a matrix, in which case the vectors are the rows
   *     of `xs`.
   *   In each case, all the vectors must have the same length and the length
   *   must be greater than or equal to the number of vectors.
   * @returns The orthogonalized and normalized vectors or matrix.
   *   Orthogonalization means that the vectors or the rows of the matrix
   *   are orthogonal (zero inner products). Normalization means that each
   *   vector or each row of the matrix has an L2 norm that equals `1`.
   */
  @doc({heading: 'Operations', subheading: 'Linear Algebra'})
  @operation
  static gramSchmidt(xs: Tensor1D[]|Tensor2D): Tensor1D[]|Tensor2D {
    let inputIsTensor2D: boolean;
    if (Array.isArray(xs)) {
      inputIsTensor2D = false;
      assert(
          xs != null && xs.length > 0,
          'Gram-Schmidt process: input must not be null, undefined, or empty');
      const dim = xs[0].shape[0];
      for (let i = 1; i < xs.length; ++i) {
        assert(
            xs[i].shape[0] === dim,
            'Gram-Schmidt: Non-unique lengths found in the input vectors: ' +
                `(${xs[i].shape[0]} vs. ${dim})`);
      }
    } else {
      inputIsTensor2D = true;
      xs = split(xs, xs.shape[0], 0).map(x => squeeze(x, [0]));
    }

    assert(
        xs.length <= xs[0].shape[0],
        `Gram-Schmidt: Number of vectors (${xs.length}) exceeds ` +
            `number of dimensions (${xs[0].shape[0]}).`);

    const ys: Tensor1D[] = [];
    const xs1d = xs as Tensor1D[];
    for (let i = 0; i < xs.length; ++i) {
      ys.push(Tracking.tidy(() => {
        let x = xs1d[i];
        if (i > 0) {
          for (let j = 0; j < i; ++j) {
            const proj = sum(ys[j].mulStrict(x)).mul(ys[j]);
            x = x.sub(proj);
          }
        }
        return x.div(norm(x, 'euclidean'));
      }));
    }

    if (inputIsTensor2D) {
      return stack(ys, 0) as Tensor2D;
    } else {
      return ys;
    }
  }
}
