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
import {MaxPoolGrad, MaxPoolGradAttrs, MaxPoolGradInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {op} from './operation';

/**
 * Computes the backprop of a 2D max pool.
 *
 * @param dy The dy error, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param input The original input image, of rank 4, of shape
 *     [batchSize, height, width, channels].
 * @param output The original output image, of rank 4, of shape
 *     [batchSize, outHeight, outWidth, channels].
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
function maxPoolGrad_(
    dy: Tensor4D|TensorLike, input: Tensor4D|TensorLike,
    output: Tensor4D|TensorLike, filterSize: [number, number]|number,
    strides: [number, number]|number, pad: 'valid'|'same'|number,
    dimRoundingMode?: 'floor'|'round'|'ceil'): Tensor4D {
  const $dy = convertToTensor(dy, 'dy', 'maxPoolGrad');
  const $input = convertToTensor(input, 'input', 'maxPoolGrad');
  const $output = convertToTensor(output, 'output', 'maxPoolGrad');

  util.assert(
      $input.rank === $dy.rank,
      () => `Rank of input (${$input.rank}) does not match rank of dy ` +
          `(${$dy.rank})`);

  util.assert(
      $dy.rank === 4,
      () => `Error in maxPoolGrad: dy must be rank 4 but got rank ` +
          `${$dy.rank}.`);
  util.assert(
      $input.rank === 4,
      () => `Error in maxPoolGrad: input must be rank 4 but got rank ` +
          `${$input.rank}.`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in maxPoolGrad: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const inputs: MaxPoolGradInputs = {dy: $dy, input: $input, output: $output};

  const attrs: MaxPoolGradAttrs = {filterSize, strides, pad, dimRoundingMode};

  // tslint:disable-next-line: no-unnecessary-type-assertion
  return ENGINE.runKernel(
             MaxPoolGrad, inputs as {} as NamedTensorMap,
             attrs as {} as NamedAttrMap) as Tensor4D;
}

export const maxPoolGrad = op({maxPoolGrad_});
