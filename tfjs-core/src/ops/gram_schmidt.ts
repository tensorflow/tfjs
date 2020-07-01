/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {ENGINE} from '../engine';
import {Tensor1D, Tensor2D} from '../tensor';
import {assert} from '../util';

import {div} from './div';
import {mul} from './mul';
import {norm} from './norm';
import {op} from './operation';
import {split} from './split';
import {squeeze} from './squeeze';
import {stack} from './stack';
import {sub} from './sub';
import {sum} from './sum';

/**
 * Gram-Schmidt orthogonalization.
 *
 * ```js
 * const x = tf.tensor2d([[1, 2], [3, 4]]);
 * let y = tf.linalg.gramSchmidt(x);
 * y.print();
 * console.log('Othogonalized:');
 * y.dot(y.transpose()).print();  // should be nearly the identity matrix.
 * console.log('First row direction maintained:');
 * const data = await y.array();
 * console.log(data[0][1] / data[0][0]);  // should be nearly 2.
 * ```
 *
 * @param xs The vectors to be orthogonalized, in one of the two following
 *   formats:
 *   - An Array of `tf.Tensor1D`.
 *   - A `tf.Tensor2D`, i.e., a matrix, in which case the vectors are the rows
 *     of `xs`.
 *   In each case, all the vectors must have the same length and the length
 *   must be greater than or equal to the number of vectors.
 * @returns The orthogonalized and normalized vectors or matrix.
 *   Orthogonalization means that the vectors or the rows of the matrix
 *   are orthogonal (zero inner products). Normalization means that each
 *   vector or each row of the matrix has an L2 norm that equals `1`.
 */
/**
 * @doc {heading:'Operations', subheading:'Linear Algebra', namespace:'linalg'}
 */
function gramSchmidt_(xs: Tensor1D[]|Tensor2D): Tensor1D[]|Tensor2D {
  let inputIsTensor2D: boolean;
  if (Array.isArray(xs)) {
    inputIsTensor2D = false;
    assert(
        xs != null && xs.length > 0,
        () => 'Gram-Schmidt process: input must not be null, undefined, or ' +
            'empty');
    const dim = xs[0].shape[0];
    for (let i = 1; i < xs.length; ++i) {
      assert(
          xs[i].shape[0] === dim,
          () =>
              'Gram-Schmidt: Non-unique lengths found in the input vectors: ' +
              `(${(xs as Tensor1D[])[i].shape[0]} vs. ${dim})`);
    }
  } else {
    inputIsTensor2D = true;
    xs = split(xs, xs.shape[0], 0).map(x => squeeze(x, [0]));
  }

  assert(
      xs.length <= xs[0].shape[0],
      () => `Gram-Schmidt: Number of vectors (${
                (xs as Tensor1D[]).length}) exceeds ` +
          `number of dimensions (${(xs as Tensor1D[])[0].shape[0]}).`);

  const ys: Tensor1D[] = [];
  const xs1d = xs;
  for (let i = 0; i < xs.length; ++i) {
    ys.push(ENGINE.tidy(() => {
      let x = xs1d[i];
      if (i > 0) {
        for (let j = 0; j < i; ++j) {
          const proj = mul(sum(mul(ys[j], x)), ys[j]);
          x = sub(x, proj);
        }
      }
      return div(x, norm(x, 'euclidean'));
    }));
  }

  if (inputIsTensor2D) {
    return stack(ys, 0) as Tensor2D;
  } else {
    return ys;
  }
}

export const gramSchmidt = op({gramSchmidt_});
