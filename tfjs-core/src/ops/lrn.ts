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

import {ENGINE} from '../engine';
import {Tensor, Tensor3D, Tensor4D} from '../tensor';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';
import {op} from './operation';

/**
 * Normalizes the activation of a local neighborhood across or within
 * channels.
 *
 * @param x The input tensor. The 4-D input tensor is treated as a 3-D array
 *     of 1D vectors (along the last dimension), and each vector is
 *     normalized independently.
 * @param depthRadius The number of adjacent channels in the 1D normalization
 *     window.
 * @param bias A constant bias term for the basis.
 * @param alpha A scale factor, usually positive.
 * @param beta An exponent.
 */
/** @doc {heading: 'Operations', subheading: 'Normalization'} */
function localResponseNormalization_<T extends Tensor3D|Tensor4D>(
    x: T|TensorLike, depthRadius = 5, bias = 1, alpha = 1, beta = 0.5): T {
  const $x = convertToTensor(x, 'x', 'localResponseNormalization');
  util.assert(
      $x.rank === 4 || $x.rank === 3,
      () => `Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${$x.rank}.`);
  util.assert(
      util.isInt(depthRadius),
      () => `Error in localResponseNormalization: depthRadius must be an ` +
          `integer but got depthRadius ${depthRadius}.`);

  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = $x.as4D(1, $x.shape[0], $x.shape[1], $x.shape[2]);
  }
  const backward = (dy: Tensor4D, saved: Tensor[]) => {
    const [x4D, y] = saved;
    return {
      x4D: () => ENGINE.runKernelFunc(
          backend => backend.LRNGrad(
              dy, x4D as Tensor4D, y as Tensor4D, depthRadius, bias, alpha,
              beta),
          {})
    };
  };
  const res = ENGINE.runKernelFunc((backend, save) => {
    const y = backend.localResponseNormalization4D(
        x4D, depthRadius, bias, alpha, beta);
    save([x4D, y]);
    return y;
  }, {x4D}, backward);
  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  } else {
    return res as T;
  }
}

export const localResponseNormalization = op({localResponseNormalization_});
