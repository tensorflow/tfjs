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
import {Tensor1D, Tensor3D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {batchNorm} from './batchnorm';
import {warnDeprecation} from './batchnorm_util';
import {op} from './operation';

/**
 * Batch normalization, strictly for 3D. For the more relaxed version, see
 * `tf.batchNorm`.
 *
 * @param x The input Tensor.
 * @param mean A mean Tensor.
 * @param variance A variance Tensor.
 * @param offset An offset Tensor.
 * @param scale A scale Tensor.
 * @param varianceEpsilon A small float number to avoid dividing by 0.
 */
function batchNorm3d_(
    x: Tensor3D|TensorLike, mean: Tensor3D|Tensor1D|TensorLike,
    variance: Tensor3D|Tensor1D|TensorLike,
    offset?: Tensor3D|Tensor1D|TensorLike, scale?: Tensor3D|Tensor1D|TensorLike,
    varianceEpsilon?: number): Tensor3D {
  const $x = convertToTensor(x, 'x', 'batchNorm');
  const $mean = convertToTensor(mean, 'mean', 'batchNorm');
  const $variance = convertToTensor(variance, 'variance', 'batchNorm');
  let $scale: Tensor3D|Tensor1D;
  if (scale != null) {
    $scale = convertToTensor(scale, 'scale', 'batchNorm');
  }
  let $offset: Tensor3D|Tensor1D;
  if (offset != null) {
    $offset = convertToTensor(offset, 'offset', 'batchNorm');
  }
  util.assert(
      $x.rank === 3,
      () => `Error in batchNorm3D: x must be rank 3 but got rank ` +
          `${$x.rank}.`);
  util.assert(
      $mean.rank === 3 || $mean.rank === 1,
      () => `Error in batchNorm3D: mean must be rank 3 or rank 1 but ` +
          `got rank ${$mean.rank}.`);
  util.assert(
      $variance.rank === 3 || $variance.rank === 1,
      () => `Error in batchNorm3D: variance must be rank 3 or rank 1 ` +
          `but got rank ${$variance.rank}.`);
  if ($scale != null) {
    util.assert(
        $scale.rank === 3 || $scale.rank === 1,
        () => `Error in batchNorm3D: scale must be rank 3 or rank 1 ` +
            `but got rank ${$scale.rank}.`);
  }
  if ($offset != null) {
    util.assert(
        $offset.rank === 3 || $offset.rank === 1,
        () => `Error in batchNorm3D: offset must be rank 3 or rank 1 ` +
            `but got rank ${$offset.rank}.`);
  }

  return batchNorm($x, $mean, $variance, $offset, $scale, varianceEpsilon);
}

/**
 * @deprecated Please use `tf.batchNorm3d` instead and note the positional
 *     argument change of scale, offset, and varianceEpsilon.
 */
function batchNormalization3d_(
    x: Tensor3D|TensorLike, mean: Tensor3D|Tensor1D|TensorLike,
    variance: Tensor3D|Tensor1D|TensorLike, varianceEpsilon = .001,
    scale?: Tensor3D|Tensor1D|TensorLike,
    offset?: Tensor3D|Tensor1D|TensorLike): Tensor3D {
  warnDeprecation();
  return batchNorm3d_(x, mean, variance, offset, scale, varianceEpsilon);
}

// todo(yassogba): Remove batchNormalization3d since it is deprecated.
export const batchNormalization3d = op({batchNormalization3d_});
export const batchNorm3d = op({batchNorm3d_});
