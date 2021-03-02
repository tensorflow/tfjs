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

import {ENGINE} from '../engine';
import {AvgPoolGrad, AvgPoolGradAttrs, AvgPoolGradInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor3D, Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {op} from './operation';
import {reshape} from './reshape';

/**
 * Computes the backprop of an 2D avg pool.
 *
 * @param dy The dy error, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param input The input image, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 */
function avgPoolGrad_<T extends Tensor3D|Tensor4D>(
    dy: T|TensorLike, input: T|TensorLike, filterSize: [number, number]|number,
    strides: [number, number]|number, pad: 'valid'|'same'|number): T {
  const $dy = convertToTensor(dy, 'dy', 'avgPoolGrad');
  const $input = convertToTensor(input, 'input', 'avgPoolGrad');

  util.assert(
      $input.rank === $dy.rank,
      () => `Rank of input (${$input.rank}) does not match rank of dy (${
          $dy.rank})`);

  let input4D = $input as Tensor4D;
  let dy4D = $dy as Tensor4D;
  let reshapedTo4D = false;

  if ($input.rank === 3) {
    reshapedTo4D = true;
    input4D =
        reshape($input, [1, $input.shape[0], $input.shape[1], $input.shape[2]]);
    dy4D = reshape($dy, [1, $dy.shape[0], $dy.shape[1], $dy.shape[2]]);
  }

  util.assert(
      dy4D.rank === 4,
      () => `Error in avgPoolGrad: dy must be rank 4 but got rank ` +
          `${dy4D.rank}.`);
  util.assert(
      input4D.rank === 4,
      () => `Error in avgPoolGrad: input must be rank 4 but got rank ` +
          `${input4D.rank}.`);

  const inputs: AvgPoolGradInputs = {dy: dy4D, input: input4D};

  const attrs: AvgPoolGradAttrs = {filterSize, strides, pad};

  // tslint:disable-next-line: no-unnecessary-type-assertion
  const res = ENGINE.runKernel(
                  AvgPoolGrad, inputs as {} as NamedTensorMap,
                  attrs as {} as NamedAttrMap) as T;

  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as T;
  }
  return res;
}

export const avgPoolGrad = op({avgPoolGrad_});
