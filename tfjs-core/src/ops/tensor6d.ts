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

import {Tensor6D} from '../tensor';
import {inferShape} from '../tensor_util_env';
import {TensorLike6D} from '../types';
import {DataType} from '../types';
import {assertNonNull} from '../util';
import {makeTensor} from './tensor_ops_util';

/**
 * Creates rank-6 `tf.Tensor` with the provided values, shape and dtype.
 *
 * The same functionality can be achieved with `tf.tensor`, but in general
 * we recommend using `tf.tensor6d` as it makes the code more readable.
 *
 *  ```js
 * // Pass a nested array.
 * tf.tensor6d([[[[[[1],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]]]]).print();
 * ```
 * ```js
 * // Pass a flat array and specify a shape.
 * tf.tensor6d([1, 2, 3, 4, 5, 6, 7, 8], [1, 1, 2, 2, 2, 1]).print();
 * ```
 *
 * @param values The values of the tensor. Can be nested array of numbers,
 *     or a flat array, or a `TypedArray`.
 * @param shape The shape of the tensor. Optional. If not provided,
 *   it is inferred from `values`.
 * @param dtype The data type.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function tensor6d(
    values: TensorLike6D,
    shape?: [number, number, number, number, number, number],
    dtype?: DataType): Tensor6D {
  assertNonNull(values);
  if (shape != null && shape.length !== 6) {
    throw new Error('tensor6d() requires shape to have six numbers');
  }
  const inferredShape = inferShape(values, dtype);
  if (inferredShape.length !== 6 && inferredShape.length !== 1) {
    throw new Error(
        'tensor6d() requires values to be number[][][][][][] or ' +
        'flat/TypedArray');
  }
  if (inferredShape.length === 1 && shape == null) {
    throw new Error(
        'tensor6d() requires shape to be provided when `values` ' +
        'are a flat array');
  }
  shape = shape ||
      inferredShape as [number, number, number, number, number, number];
  return makeTensor(values, shape, inferredShape, dtype) as Tensor6D;
}
