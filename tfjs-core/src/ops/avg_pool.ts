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

import {ENGINE, ForwardFunc} from '../engine';
import {AvgPool, AvgPoolAttrs, AvgPoolInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor3D, Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {cast, reshape} from './array_ops';
import * as conv_util from './conv_util';
import {op} from './operation';

/**
 * Computes the 2D average pooling of an image.
 *
 * @param x The input tensor, of rank 4 or rank 3 of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param pad The type of padding algorithm:
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *    - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *         https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dimRoundingMode The rounding mode used when computing output
 *     dimensions if pad is a number. If none is provided, it will not round
 *     and error if the output is of fractional size.
 */
function avgPool_<T extends Tensor3D|Tensor4D>(
    x: T|TensorLike, filterSize: [number, number]|number,
    strides: [number, number]|number, pad: 'valid'|'same'|number,
    dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  const $x = convertToTensor(x, 'x', 'avgPool', 'float32');
  const dilations = 1;

  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in avgPool: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);

  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;
  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }

  util.assert(
      x4D.rank === 4,
      () => `Error in avgPool: x must be rank 4 but got rank ${x4D.rank}.`);

  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in avgPool: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    const convInfo = conv_util.computePool2DInfo(
        x4D.shape, filterSize, strides, 1 /* dilations */, pad,
        dimRoundingMode);

    save([x4D]);

    if (convInfo.filterWidth === 1 && convInfo.filterHeight === 1 &&
        util.arraysEqual(convInfo.inShape, convInfo.outShape)) {
      return x4D.clone();
    }

    return backend.avgPool(x4D, convInfo);
  };

  const inputs: AvgPoolInputs = {x: x4D};

  const attrs: AvgPoolAttrs = {filterSize, strides, pad, dimRoundingMode};

  let res = ENGINE.runKernelFunc(
      forward, inputs as {} as NamedTensorMap, null /* grad */, AvgPool,
      attrs as {} as NamedAttrMap);

  res = cast(res, $x.dtype);

  if (reshapedTo4D) {
    return res.as3D(res.shape[1], res.shape[2], res.shape[3]) as T;
  }

  return res as T;
}

export const avgPool = op({avgPool_});
