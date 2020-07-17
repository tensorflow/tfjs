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

import {ENGINE} from '../engine';
import {Tensor} from '../tensor';
import {DataType, Rank, ShapeMap} from '../types';
import {makeZerosTypedArray, sizeFromShape} from '../util';

import {complex} from './complex';

/**
 * Creates a `tf.Tensor` with all elements set to 0.
 *
 * ```js
 * tf.zeros([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param dtype The type of an element in the resulting tensor. Can
 *     be 'float32', 'int32' or 'bool'. Defaults to 'float'.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
export function zeros<R extends Rank>(
    shape: ShapeMap[R], dtype: DataType = 'float32'): Tensor<R> {
  if (dtype === 'complex64') {
    const real = zeros(shape, 'float32');
    const imag = zeros(shape, 'float32');
    return complex(real, imag);
  }
  const values = makeZerosTypedArray(sizeFromShape(shape), dtype);
  return ENGINE.makeTensor(values, shape, dtype) as Tensor<R>;
}
