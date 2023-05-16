/**
 * @license
 * Copyright 2023 Google LLC.
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
import {Rank, ShapeMap} from '../types';
import {op} from './operation';
import {randomUniform} from './random_uniform';

/**
 * Creates a `tf.Tensor` with integers sampled from a uniform distribution.
 *
 * The generated values are uniform integers in the range [minval, maxval). The
 * lower bound minval is included in the range, while the upper bound maxval is
 * excluded.
 *
 * ```js
 * tf.randomUniformInt([2, 2], 0, 10).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param minval Inclusive lower bound on the generated integers.
 * @param maxval Exclusive upper bound on the generated integers.
 * @param seed An optional int. Defaults to 0. If seed is set to be non-zero,
 *   the random number generator is seeded by the given seed. Otherwise, it is
 *   seeded by a random seed.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function randomUniformInt_<R extends Rank>(
  shape: ShapeMap[R], minval: number, maxval: number,
    seed?: number|string): Tensor<R> {
  // TODO(mattsoulanille): Handle optional seed2 input.
  return randomUniform(shape, minval, maxval, 'int32', seed);
}

export const randomUniformInt = /* @__PURE__ */ op({randomUniformInt_});
