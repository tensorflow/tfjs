/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

import {ENV} from '../environment';
import {Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D} from '../tensor';
import {convertToTensorArray} from '../tensor_util_env';
import {TensorLike} from '../types';
import {assert, sizeFromShape} from '../util';
import {parseAxisParam} from './axis_util';
import * as concat_util from './concat_util';
import {op} from './operation';

/**
 * Concatenates a list of `Tensor1D`s along an axis. See `concat` for details.
 *
 * For example, if:
 * A: shape(3) = |r1, g1, b1|
 * B: shape(2) = |r2, g2|
 * C = tf.concat1d([A, B]) == |r1, g1, b1, r2, g2|
 *
 * @param tensors A list of `Tensor`s to concatenate.
 * @return The concatenated array.
 */
function concat1d_(tensors: Tensor1D[]|TensorLike[]): Tensor1D {
  return concat(tensors, 0 /* axis */);
}

/**
 * Concatenates a list of `Tensor2D`s along an axis. See `concat` for details.
 *
 * For example, if:
 * A: shape(2, 3) = | r1, g1, b1 |
 *                  | r2, g2, b2 |
 *
 * B: shape(2, 3) = | r3, g3, b3 |
 *                  | r4, g4, b4 |
 *
 * C = tf.concat2d([A, B], axis)
 *
 * if axis = 0:
 * C: shape(4, 3) = | r1, g1, b1 |
 *                  | r2, g2, b2 |
 *                  | r3, g3, b3 |
 *                  | r4, g4, b4 |
 *
 * if axis = 1:
 * C = shape(2, 6) = | r1, g1, b1, r3, g3, b3 |
 *                   | r2, g2, b2, r4, g4, b4 |
 *
 *
 * @param tensors A list of `Tensor`s to concatenate.
 * @param axis The axis to concatenate along.
 * @return The concatenated array.
 */
function concat2d_(tensors: Tensor2D[]|TensorLike[], axis: number): Tensor2D {
  return concat(tensors, axis);
}

/**
 * Concatenates a list of `Tensor3D`s along an axis. See `concat` for details.
 *
 * For example, if:
 * A: shape(2, 1, 3) = | r1, g1, b1 |
 *                     | r2, g2, b2 |
 *
 * B: shape(2, 1, 3) = | r3, g3, b3 |
 *                     | r4, g4, b4 |
 *
 * C = tf.concat3d([A, B], axis)
 *
 * if axis = 0:
 * C: shape(4, 1, 3) = | r1, g1, b1 |
 *                     | r2, g2, b2 |
 *                     | r3, g3, b3 |
 *                     | r4, g4, b4 |
 *
 * if axis = 1:
 * C: shape(2, 2, 3) = | r1, g1, b1, r3, g3, b3 |
 *                     | r2, g2, b2, r4, g4, b4 |
 *
 * if axis = 2:
 * C = shape(2, 1, 6) = | r1, g1, b1, r3, g3, b3 |
 *                      | r2, g2, b2, r4, g4, b4 |
 *
 * @param tensors A list of `Tensor`s to concatenate.
 * @param axis The axis to concate along.
 * @return The concatenated array.
 */
function concat3d_(tensors: Tensor3D[]|TensorLike[], axis: number): Tensor3D {
  return concat(tensors, axis);
}

/**
 * Concatenates a list of `Tensor4D`s along an axis. See `concat` for details.
 *
 * @param tensors A list of `Tensor`s to concatenate.
 * @param axis The axis to concate along.
 * @return The concatenated array.
 */
function concat4d_(tensors: Tensor4D[]|TensorLike[], axis: number): Tensor4D {
  return concat(tensors, axis);
}

/**
 * Concatenates a list of `Tensor`s along a given axis.
 *
 * The tensors ranks and types must match, and their sizes must match in all
 * dimensions except `axis`.
 *
 * Also available are stricter rank-specific methods that assert that
 * `tensors` are of the given rank:
 *   - `tf.concat1d`
 *   - `tf.concat2d`
 *   - `tf.concat3d`
 *   - `tf.concat4d`
 *
 * Except `tf.concat1d` (which does not have axis param), all methods have
 * same signature as this method.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * a.concat(b).print();  // or a.concat(b)
 * ```
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * const c = tf.tensor1d([5, 6]);
 * tf.concat([a, b, c]).print();
 * ```
 *
 * ```js
 * const a = tf.tensor2d([[1, 2], [10, 20]]);
 * const b = tf.tensor2d([[3, 4], [30, 40]]);
 * const axis = 1;
 * tf.concat([a, b], axis).print();
 * ```
 * @param tensors A list of tensors to concatenate.
 * @param axis The axis to concate along. Defaults to 0 (the first dim).
 */
/** @doc {heading: 'Tensors', subheading: 'Slicing and Joining'} */
function concat_<T extends Tensor>(tensors: T[]|TensorLike[], axis = 0): T {
  assert(tensors.length >= 1, 'Pass at least one tensor to concat');
  const $tensors = convertToTensorArray(tensors, 'tensors', 'concat');

  let result = $tensors[0] as T;
  if ($tensors.length === 1) {
    return result;
  }
  const axes = parseAxisParam(axis, result.shape);

  for (let i = 1; i < $tensors.length; ++i) {
    result = concat2Tensors(result, $tensors[i], axes[0]) as T;
  }
  return result;
}

function concat2Tensors<T extends Tensor>(a: T, b: T, axis: number): T {
  concat_util.assertParams(a.shape, b.shape, axis);
  const outShape = concat_util.computeOutShape(a.shape, b.shape, axis);

  // Do the reshape.
  const a2D = a.as2D(-1, sizeFromShape(a.shape.slice(axis)));
  const b2D = b.as2D(-1, sizeFromShape(b.shape.slice(axis)));
  // Concats 2d tensors along axis=1. See comments in MathBackend.concat().
  const {aBegin, aSize, bBegin, bSize} =
      concat_util.computeGradientSliceShapes(a2D.shape, b2D.shape);
  const der = (dy: Tensor2D) => {
    return {a: () => dy.slice(aBegin, aSize), b: () => dy.slice(bBegin, bSize)};
  };
  const res = ENV.engine.runKernel(
      backend => backend.concat(a2D, b2D), {a: a2D, b: b2D}, der);
  return res.reshape(outShape) as T;
}

export const concat = op({concat_});
export const concat1d = op({concat1d_});
export const concat2d = op({concat2d_});
export const concat3d = op({concat3d_});
export const concat4d = op({concat4d_});
