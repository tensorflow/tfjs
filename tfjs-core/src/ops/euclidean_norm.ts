/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {Tensor} from '../tensor';
import {TensorLike} from '../types';

import {norm} from './norm';
import {op} from './operation';

/**
 * Computes the euclidean norm of scalar, vectors, and matrices.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 *
 * x.euclideanNorm().print();  // or tf.euclideanNorm(x)
 * ```
 *
 * @param x The input array.
 * @param axis Optional. If axis is null (the default), the input is
 * considered a vector and a single vector norm is computed over the entire
 * set of values in the Tensor, i.e. euclideanNorm(x) is equivalent
 * to euclideanNorm(x.reshape([-1])). If axis is a integer, the input
 * is considered a batch of vectors, and axis determines the axis in x
 * over which to compute vector norms. If axis is a 2-tuple of integer it is
 * considered a batch of matrices and axis determines the axes in NDArray
 * over which to compute a matrix norm.
 * @param keepDims Optional. If true, the norm have the same dimensionality
 * as the input.
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function euclideanNorm_(
    x: Tensor|TensorLike, axis: number|number[] = null,
    keepDims = false): Tensor {
  return norm(x, 'euclidean', axis, keepDims);
}

export const euclideanNorm = op({euclideanNorm_});
