/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
import {Tensor, TensorBuffer} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

/**
 * Computes the difference between two lists of numbers.
 *
 * Given a Tensor `x` and a Tensor `y`, this operation returns a Tensor `out`
 * that represents all values that are in `x` but not in `y`. The returned
 * Tensor `out` is sorted in the same order that the numbers appear in `x`
 * (duplicates are preserved). This operation also returns a Tensor indices that
 * represents the position of each out element in `x`. In other words:
 *
 * `out[i] = x[idx[i]] for i in [0, 1, ..., out.length - 1]`
 *
 * ```js
 * const x = [1, 2, 3, 4, 5, 6];
 * const y = [1, 3, 5];
 *
 * const [out, indices] = await tf.setdiff1dAsync(x, y);
 * out.print(); // [2, 4, 6]
 * indices.print(); // [1, 3, 5]
 * ```
 *
 * @param x 1-D Tensor. Values to keep.
 * @param y 1-D Tensor. Must have the same type as x. Values to exclude in the
 *     output.
 * @returns Promise of Tensor tuple [out, indices].
 *  out: Tensor with the same type as x.
 *  indices: A Tensor of type int32.
 */
/** @doc {heading: 'Tensors', subheading: 'Transformations'} */
async function setdiff1dAsync_(
    x: Tensor|TensorLike, y: Tensor|TensorLike): Promise<[Tensor, Tensor]> {
  const $x = convertToTensor(x, 'x', 'setdiff1d');
  const $y = convertToTensor(y, 'y', 'setdiff1d');

  util.assert(
      $x.dtype === $y.dtype,
      () => `x and y should have the same dtype, but got x (${
          $x.dtype}) and y (${$y.dtype}).`);

  util.assert(
      $x.rank === 1, () => `x should be 1D tensor, but got x (${$x.shape}).`);

  util.assert(
      $y.rank === 1, () => `y should be 1D tensor, but got y (${$y.shape}).`);

  const xVals = await $x.data();
  const yVals = await $y.data();
  const ySet = new Set(yVals);

  let outputSize = 0;
  for (let i = 0; i < xVals.length; i++) {
    if (!ySet.has(xVals[i])) {
      outputSize++;
    }
  }

  const buffer = new TensorBuffer([outputSize], $x.dtype);
  const indices = new TensorBuffer([outputSize], 'int32');
  for (let i = 0, p = 0; i < xVals.length; i++) {
    if (!ySet.has(xVals[i])) {
      buffer.values[p] = xVals[i];
      indices.values[p] = i;
      p++;
    }
  }
  return [buffer.toTensor(), indices.toTensor()];
}
export const setdiff1dAsync = setdiff1dAsync_;
