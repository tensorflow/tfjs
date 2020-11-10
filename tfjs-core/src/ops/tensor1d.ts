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

import {Tensor1D} from '../tensor';
import {inferShape} from '../tensor_util_env';
import {TensorLike1D} from '../types';
import {DataType} from '../types';
import {assertNonNull} from '../util';
import {makeTensor} from './tensor_ops_util';

/**
 * Creates rank-1 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor1d` as it makes the code more readable.
 *
 * ```js
 * tf.tensor1d([1, 2, 3]).print();
 * ```
 *
 * @param values The values of the tensor. Can be array of numbers,
 *     or a `TypedArray`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function tensor1d(values: TensorLike1D, dtype?: DataType): Tensor1D {
  assertNonNull(values);
  const inferredShape = inferShape(values, dtype);
  if (inferredShape.length !== 1) {
    throw new Error('tensor1d() requires values to be a flat/TypedArray');
  }
  const shape: number[] = null;
  return makeTensor(values, shape, inferredShape, dtype) as Tensor1D;
}
