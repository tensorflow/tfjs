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
import {ENGINE} from '../../engine';
import {dispose} from '../../globals';
import {Tensor, Tensor2D} from '../../tensor';
import {assert} from '../../util';

import {clone} from '../clone';
import {concat} from '../concat';
import {div} from '../div';
import {eye} from '../eye';
import {greater} from '../greater';
import {matMul} from '../mat_mul';
import {mul} from '../mul';
import {neg} from '../neg';
import {norm} from '../norm';
import {op} from '../operation';
import {reshape} from '../reshape';
import {slice} from '../slice';
import {stack} from '../stack';
import {sub} from '../sub';
import {tensor2d} from '../tensor2d';
import {transpose} from '../transpose';
import {unstack} from '../unstack';
import {where} from '../where';

/**
 * Compute QR decomposition of m-by-n matrix using Householder transformation.
 *
 * Implementation based on
 *   [http://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf]
 * (http://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf)
 *
 * ```js
 * const a = tf.tensor2d([[1, 2], [3, 4]]);
 * let [q, r] = tf.linalg.qr(a);
 * console.log('Q');
 * q.print();
 * console.log('R');
 * r.print();
 * console.log('Orthogonalized');
 * q.dot(q.transpose()).print()  // should be nearly the identity matrix.
 * console.log('Reconstructed');
 * q.dot(r).print(); // should be nearly [[1, 2], [3, 4]];
 * ```
 *
 * @param x The `tf.Tensor` to be QR-decomposed. Must have rank >= 2. Suppose
 *   it has the shape `[..., M, N]`.
 * @param fullMatrices An optional boolean parameter. Defaults to `false`.
 *   If `true`, compute full-sized `Q`. If `false` (the default),
 *   compute only the leading N columns of `Q` and `R`.
 * @returns An `Array` of two `tf.Tensor`s: `[Q, R]`. `Q` is a unitary matrix,
 *   i.e., its columns all have unit norm and are mutually orthogonal.
 *   If `M >= N`,
 *     If `fullMatrices` is `false` (default),
 *       - `Q` has a shape of `[..., M, N]`,
 *       - `R` has a shape of `[..., N, N]`.
 *     If `fullMatrices` is `true` (default),
 *       - `Q` has a shape of `[..., M, M]`,
 *       - `R` has a shape of `[..., M, N]`.
 *   If `M < N`,
 *     - `Q` has a shape of `[..., M, M]`,
 *     - `R` has a shape of `[..., M, N]`.
 * @throws If the rank of `x` is less than 2.
 *
 * @doc {heading:'Operations',
 *       subheading:'Linear Algebra',
 *       namespace:'linalg'}
 */
function qr_(x: Tensor, fullMatrices = false): [Tensor, Tensor] {
  assert(
      x.rank >= 2,
      () => `qr() requires input tensor to have a rank >= 2, but got rank ${
          x.rank}`);

  if (x.rank === 2) {
    return qr2d(x as Tensor2D, fullMatrices);
  } else {
    // Rank > 2.
    // TODO(cais): Below we split the input into individual 2D tensors,
    //   perform QR decomposition on them and then stack the results back
    //   together. We should explore whether this can be parallelized.
    const outerDimsProd = x.shape.slice(0, x.shape.length - 2)
                              .reduce((value, prev) => value * prev);
    const x2ds = unstack(
        reshape(
            x,
            [
              outerDimsProd, x.shape[x.shape.length - 2],
              x.shape[x.shape.length - 1]
            ]),
        0);
    const q2ds: Tensor2D[] = [];
    const r2ds: Tensor2D[] = [];
    x2ds.forEach(x2d => {
      const [q2d, r2d] = qr2d(x2d as Tensor2D, fullMatrices);
      q2ds.push(q2d);
      r2ds.push(r2d);
    });
    const q = reshape(stack(q2ds, 0), x.shape);
    const r = reshape(stack(r2ds, 0), x.shape);
    return [q, r];
  }
}

function qr2d(x: Tensor2D, fullMatrices = false): [Tensor2D, Tensor2D] {
  return ENGINE.tidy(() => {
    assert(
        x.shape.length === 2,
        () => `qr2d() requires a 2D Tensor, but got a ${
            x.shape.length}D Tensor.`);

    const m = x.shape[0];
    const n = x.shape[1];

    let q = eye(m);    // Orthogonal transform so far.
    let r = clone(x);  // Transformed matrix so far.

    const one2D = tensor2d([[1]], [1, 1]);
    let w: Tensor2D = clone(one2D);

    const iters = m >= n ? n : m;
    for (let j = 0; j < iters; ++j) {
      // This tidy within the for-loop ensures we clean up temporary
      // tensors as soon as they are no longer needed.
      const rTemp = r;
      const wTemp = w;
      const qTemp = q;
      [w, r, q] = ENGINE.tidy((): [Tensor2D, Tensor2D, Tensor2D] => {
        // Find H = I - tau * w * w', to put zeros below R(j, j).
        const rjEnd1 = slice(r, [j, j], [m - j, 1]);
        const normX = norm(rjEnd1);
        const rjj = slice(r, [j, j], [1, 1]);

        // The sign() function returns 0 on 0, which causes division by zero.
        const s = where(greater(rjj, 0), tensor2d([[-1]]), tensor2d([[1]]));

        const u1 = sub(rjj, mul(s, normX));
        const wPre = div(rjEnd1, u1);
        if (wPre.shape[0] === 1) {
          w = clone(one2D);
        } else {
          w = concat(
              [
                one2D,
                slice(wPre, [1, 0], [wPre.shape[0] - 1, wPre.shape[1]]) as
                    Tensor2D
              ],
              0);
        }
        const tau = neg(div(matMul(s, u1), normX)) as Tensor2D;

        // -- R := HR, Q := QH.
        const rjEndAll = slice(r, [j, 0], [m - j, n]);
        const tauTimesW: Tensor2D = mul(tau, w);
        const wT: Tensor2D = transpose(w);
        if (j === 0) {
          r = sub(rjEndAll, matMul(tauTimesW, matMul(wT, rjEndAll)));
        } else {
          const rTimesTau: Tensor2D =
              sub(rjEndAll, matMul(tauTimesW, matMul(wT, rjEndAll)));
          r = concat([slice(r, [0, 0], [j, n]), rTimesTau], 0);
        }
        const tawTimesWT: Tensor2D = transpose(tauTimesW);
        const qAllJEnd = slice(q, [0, j], [m, q.shape[1] - j]);
        if (j === 0) {
          q = sub(qAllJEnd, matMul(matMul(qAllJEnd, w), tawTimesWT));
        } else {
          const qTimesTau: Tensor2D =
              sub(qAllJEnd, matMul(matMul(qAllJEnd, w), tawTimesWT));
          q = concat([slice(q, [0, 0], [m, j]), qTimesTau], 1);
        }
        return [w, r, q];
      });
      dispose([rTemp, wTemp, qTemp]);
    }

    if (!fullMatrices && m > n) {
      q = slice(q, [0, 0], [m, n]);
      r = slice(r, [0, 0], [n, n]);
    }

    return [q, r];
  }) as [Tensor2D, Tensor2D];
}

export const qr = op({qr_});
