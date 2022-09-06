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

import {Tensor} from '../tensor';
import {Rank, ShapeMap} from '../types';

import {buffer} from './buffer';
import {op} from './operation';
import {RandGamma} from './rand_util';

/**
 * Creates a `tf.Tensor` with values sampled from a gamma distribution.
 *
 * ```js
 * tf.randomGamma([2, 2], 1).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param alpha The shape parameter of the gamma distribution.
 * @param beta The inverse scale parameter of the gamma distribution. Defaults
 *     to 1.
 * @param dtype The data type of the output. Defaults to float32.
 * @param seed The seed for the random number generator.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function randomGamma_<R extends Rank>(
    shape: ShapeMap[R], alpha: number, beta = 1,
    dtype: 'float32'|'int32' = 'float32', seed?: number): Tensor<R> {
  if (beta == null) {
    beta = 1;
  }
  if (dtype == null) {
    dtype = 'float32';
  }
  if (dtype !== 'float32' && dtype !== 'int32') {
    throw new Error(`Unsupported data type ${dtype}`);
  }
  const rgamma = new RandGamma(alpha, beta, dtype, seed);
  const res = buffer(shape, dtype);
  for (let i = 0; i < res.values.length; i++) {
    res.values[i] = rgamma.nextValue();
  }
  return res.toTensor();
}

export const randomGamma = op({randomGamma_});
