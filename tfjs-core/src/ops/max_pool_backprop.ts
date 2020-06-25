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

import {ENGINE, ForwardFunc} from '../engine';
import {MaxPoolBackprop, MaxPoolBackpropAttrs, MaxPoolBackpropInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import * as conv_util from './conv_util';
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
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. The
 *     rounding mode used when computing output dimensions if pad is a
 *     number. If none is provided, it will not round and error if the output
 *     is of fractional size.
 */
function maxPoolBackprop_(
    dy: Tensor4D|TensorLike, input: Tensor4D|TensorLike,
    output: Tensor4D|TensorLike, filterSize: [number, number]|number,
    strides: [number, number]|number, pad: 'valid'|'same'|number,
    dimRoundingMode?: 'floor'|'round'|'ceil'): Tensor4D {
  const $dy = convertToTensor(dy, 'dy', 'maxPoolBackprop');
  const $input = convertToTensor(input, 'input', 'maxPoolBackprop');
  const $output = convertToTensor(output, 'output', 'maxPoolBackprop');

  util.assert(
      $input.rank === $dy.rank,
      () => `Rank of input (${$input.rank}) does not match rank of dy ` +
          `(${$dy.rank})`);

  util.assert(
      $dy.rank === 4,
      () => `Error in maxPoolBackprop: dy must be rank 4 but got rank ` +
          `${$dy.rank}.`);
  util.assert(
      $input.rank === 4,
      () => `Error in maxPoolBackprop: input must be rank 4 but got rank ` +
          `${$input.rank}.`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in maxPoolBackprop: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const forward: ForwardFunc<Tensor> = backend => {
    const convInfo = conv_util.computePool2DInfo(
        $input.shape, filterSize, strides, 1 /* dilations */, pad,
        dimRoundingMode);

    return backend.maxPoolBackprop($dy, $input, $output, convInfo);
  };

  const inputs:
      MaxPoolBackpropInputs = {dy: $dy, input: $input, output: $output};

  const attrs:
      MaxPoolBackpropAttrs = {filterSize, strides, pad, dimRoundingMode};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null, MaxPoolBackprop,
             attrs as {} as NamedAttrMap) as Tensor4D;
}

export const maxPoolBackprop = op({maxPoolBackprop_});
