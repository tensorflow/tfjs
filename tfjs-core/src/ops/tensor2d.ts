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

import {Tensor2D} from '../tensor';
import {inferShape} from '../tensor_util_env';
import {TensorLike2D} from '../types';
import {DataType} from '../types';
import {assertNonNull} from '../util';
import {makeTensor} from './tensor_ops_util';

/**
 * Creates rank-2 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor2d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor2d([[1, 2], [3, 4]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor2d([1, 2, 3, 4], [2, 2]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. If not provided, it is inferred from
 *     `values`.
 * @param dtype The data type.
 */
/** @doc {heading: 'Tensors', subheading: 'Creation'} */
export function tensor2d(
    values: TensorLike2D, shape?: [number, number],
    dtype?: DataType): Tensor2D {
  assertNonNull(values);
  if (shape != null && shape.length !== 2) {
    throw new Error('tensor2d() requires shape to have two numbers');
  }
  const inferredShape = inferShape(values, dtype);
  if (inferredShape.length !== 2 && inferredShape.length !== 1) {
    throw new Error(
        'tensor2d() requires values to be number[][] or flat/TypedArray');
  }
  if (inferredShape.length === 1 && shape == null) {
    throw new Error(
        'tensor2d() requires shape to be provided when `values` ' +
        'are a flat/TypedArray');
  }
  return makeTensor(values, shape, inferredShape, dtype) as Tensor2D;
}
