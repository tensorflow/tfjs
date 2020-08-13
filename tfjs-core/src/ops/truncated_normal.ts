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
import {DataType, Rank, ShapeMap} from '../types';

import {buffer} from './buffer';
import {op} from './operation';
import {MPRandGauss} from './rand_util';

/**
 * Creates a `tf.Tensor` with values sampled from a truncated normal
 * distribution.
 *
 * ```js
 * tf.truncatedNormal([2, 2]).print();
 * ```
 *
 * The generated values follow a normal distribution with specified mean and
 * standard deviation, except that values whose magnitude is more than 2
 * standard deviations from the mean are dropped and re-picked.
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param mean The mean of the normal distribution.
 * @param stdDev The standard deviation of the normal distribution.
 * @param dtype The data type of the output tensor.
 * @param seed The seed for the random number generator.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
function truncatedNormal_<R extends Rank>(
    shape: ShapeMap[R], mean = 0, stdDev = 1, dtype?: 'float32'|'int32',
    seed?: number): Tensor<R> {
  if (dtype != null && (dtype as DataType) === 'bool') {
    throw new Error(`Unsupported data type $ { dtype }`);
  }
  const randGauss =
      new MPRandGauss(mean, stdDev, dtype, true /* truncated */, seed);
  const res = buffer(shape, dtype);
  for (let i = 0; i < res.values.length; i++) {
    res.values[i] = randGauss.nextValue();
  }
  return res.toTensor();
}

export const truncatedNormal = op({truncatedNormal_});
