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
import {convertToTensor} from '../tensor_util_env';
import {Rank, ShapeMap, TensorLike} from '../types';
import {arraysEqualWithNull} from '../util_base';

import {op} from './operation';

/**
 * Updates a tensor and checks at runtime that the shape holds.
 *
 * Given an input tensor, returns a new tensor with the same values as the
 * input tensor with shape `shape`.
 *
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 * tf.ensureShape(x, [4]).print();
 * ```
 *
 * @param x The input tensor to be ensured.
 * @param shape A TensorShape representing the shape of this tensor, a list, a
 *     tuple, or None.
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function ensureShape_<R extends Rank>(
    x: Tensor|TensorLike, shape: ShapeMap[R]): Tensor {
  const $x = convertToTensor(x, 'x', 'ensure_shape', 'string_or_numeric');

  if (!arraysEqualWithNull($x.shape, shape)) {
    throw new Error(`Invalid argument error. Shape of tensor ${
        x} is not compatible with expected shape ${shape}`);
  }

  return $x;
}
export const ensureShape = /* @__PURE__ */ op({ensureShape_});
