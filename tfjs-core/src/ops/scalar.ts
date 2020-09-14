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

import {Scalar} from '../tensor';
import {DataType} from '../types';
import {isTypedArray} from '../util';
import {makeTensor} from './tensor_ops_util';

/**
 * Creates rank-0 `tf.Tensor` (scalar) with the provided value and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.scalar` as it makes the code more readable.
 *
 * ```js
 * tf.scalar(3.14).print();
 * ```
 *
 * @param value The value of the scalar.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function scalar(
    value: number|boolean|string|Uint8Array, dtype?: DataType): Scalar {
  if (((isTypedArray(value) && dtype !== 'string') || Array.isArray(value)) &&
      dtype !== 'complex64') {
    throw new Error(
        'Error creating a new Scalar: value must be a primitive ' +
        '(number|boolean|string)');
  }
  if (dtype === 'string' && isTypedArray(value) &&
      !(value instanceof Uint8Array)) {
    throw new Error(
        'When making a scalar from encoded string, ' +
        'the value must be `Uint8Array`.');
  }
  const shape: number[] = [];
  const inferredShape: number[] = [];
  return makeTensor(value, shape, inferredShape, dtype) as Scalar;
}
