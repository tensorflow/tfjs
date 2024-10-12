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
import {Tensor1D, Tensor5D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {batchNorm} from './batchnorm';
import {op} from './operation';

/**
 * Batch normalization, strictly for 5D. For the more relaxed version, see
 * `tf.batchNorm`.
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 */
function batchNorm5d_(
    x: Tensor5D|TensorLike, mean: Tensor5D|Tensor1D|TensorLike,
    variance: Tensor5D|Tensor1D|TensorLike,
    offset?: Tensor5D|Tensor1D|TensorLike, scale?: Tensor5D|Tensor1D|TensorLike,
    varianceEpsilon?: number): Tensor5D {
  const $x = convertToTensor(x, 'x', 'batchNorm');
  const $mean = convertToTensor(mean, 'mean', 'batchNorm');
  const $variance = convertToTensor(variance, 'variance', 'batchNorm');
  let $scale: Tensor5D|Tensor1D;
  if (scale != null) {
    $scale = convertToTensor(scale, 'scale', 'batchNorm');
  }
  let $offset: Tensor5D|Tensor1D;
  if (offset != null) {
    $offset = convertToTensor(offset, 'offset', 'batchNorm');
  }
  util.assert(
      $x.rank === 5,
      () => `Error in batchNorm5D: x must be rank 5 but got rank ` +
          `${$x.rank}.`);
  util.assert(
      $mean.rank === 5 || $mean.rank === 1,
      () => `Error in batchNorm5D: mean must be rank 5 or rank 1 but ` +
          `got rank ${$mean.rank}.`);
  util.assert(
      $variance.rank === 5 || $variance.rank === 1,
      () => `Error in batchNorm5D: variance must be rank 5 or rank 1 ` +
          `but got rank ${$variance.rank}.`);
  if ($scale != null) {
    util.assert(
        $scale.rank === 5 || $scale.rank === 1,
        () => `Error in batchNorm5D: scale must be rank 5 or rank 1 ` +
            `but got rank ${$scale.rank}.`);
  }
  if ($offset != null) {
    util.assert(
        $offset.rank === 5 || $offset.rank === 1,
        () => `Error in batchNorm5D: offset must be rank 5 or rank 1 ` +
            `but got rank ${$offset.rank}.`);
  }
  return batchNorm($x, $mean, $variance, $offset, $scale, varianceEpsilon);
}

export const batchNorm5d = op({batchNorm5d_});
