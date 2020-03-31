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
import {Tensor1D, Tensor4D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {batchNorm} from './batchnorm';
import {warnDeprecation} from './batchnorm_util';
import {op} from './operation';

/**
 * Batch normalization, strictly for 4D. For the more relaxed version, see
 * `tf.batchNorm`.
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 */
function batchNorm4d_(
    x: Tensor4D|TensorLike, mean: Tensor4D|Tensor1D|TensorLike,
    variance: Tensor4D|Tensor1D|TensorLike,
    offset?: Tensor4D|Tensor1D|TensorLike, scale?: Tensor4D|Tensor1D|TensorLike,
    varianceEpsilon?: number): Tensor4D {
  const $x = convertToTensor(x, 'x', 'batchNorm');
  const $mean = convertToTensor(mean, 'mean', 'batchNorm');
  const $variance = convertToTensor(variance, 'variance', 'batchNorm');
  let $scale: Tensor4D|Tensor1D;
  if (scale != null) {
    $scale = convertToTensor(scale, 'scale', 'batchNorm');
  }
  let $offset: Tensor4D|Tensor1D;
  if (offset != null) {
    $offset = convertToTensor(offset, 'offset', 'batchNorm');
  }
  util.assert(
      $x.rank === 4,
      () => `Error in batchNorm4D: x must be rank 4 but got rank ` +
          `${$x.rank}.`);
  util.assert(
      $mean.rank === 4 || $mean.rank === 1,
      () => `Error in batchNorm4D: mean must be rank 4 or rank 1 but ` +
          `got rank ${$mean.rank}.`);
  util.assert(
      $variance.rank === 4 || $variance.rank === 1,
      () => `Error in batchNorm4D: variance must be rank 4 or rank 1 ` +
          `but got rank ${$variance.rank}.`);
  if ($scale != null) {
    util.assert(
        $scale.rank === 4 || $scale.rank === 1,
        () => `Error in batchNorm4D: scale must be rank 4 or rank 1 ` +
            `but got rank ${$scale.rank}.`);
  }
  if ($offset != null) {
    util.assert(
        $offset.rank === 4 || $offset.rank === 1,
        () => `Error in batchNorm4D: offset must be rank 4 or rank 1 ` +
            `but got rank ${$offset.rank}.`);
  }
  return batchNorm($x, $mean, $variance, $offset, $scale, varianceEpsilon);
}

/**
 * @deprecated Please use `tf.batchNorm4d` instead and note the positional
 *     argument change of scale, offset, and varianceEpsilon.
 */
function batchNormalization4d_(
    x: Tensor4D|TensorLike, mean: Tensor4D|Tensor1D|TensorLike,
    variance: Tensor4D|Tensor1D|TensorLike, varianceEpsilon = .001,
    scale?: Tensor4D|Tensor1D|TensorLike,
    offset?: Tensor4D|Tensor1D|TensorLike): Tensor4D {
  warnDeprecation();
  return batchNorm4d_(x, mean, variance, offset, scale, varianceEpsilon);
}

// todo(yassogba): Remove batchNormalization4d since it is deprecated.
export const batchNormalization4d = op({batchNormalization4d_});
export const batchNorm4d = op({batchNorm4d_});
