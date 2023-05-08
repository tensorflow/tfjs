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
import {DataType, Rank, ShapeMap} from '../types';

import {op} from './operation';
import {randomNormal} from './random_normal';

/**
 * Creates a `tf.Tensor` with values sampled from a normal distribution.
 *
 * The generated values will have mean 0 and standard deviation 1.
 *
 * ```js
 * tf.randomStandardNormal([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The data type of the output.
 * @param seed The seed for the random number generator.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function randomStandardNormal_<R extends Rank>(
    shape: ShapeMap[R], dtype?: 'float32'|'int32', seed?: number): Tensor<R> {
  if (dtype != null && (dtype as DataType) === 'bool') {
    throw new Error(`Unsupported data type ${dtype}`);
  }
  return randomNormal(shape, 0, 1, dtype, seed);
}

export const randomStandardNormal = /* @__PURE__ */ op({randomStandardNormal_});
