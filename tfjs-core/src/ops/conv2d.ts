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
import {Conv2D, Conv2DAttrs, Conv2DInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor3D, Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import {reshape} from './array_ops';
import * as conv_util from './conv_util';
import {op} from './operation';

/**
 * Computes a 2D convolution over the input x.
 *
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter, rank 4, of shape
 *     `[filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels].
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in atrous convolution. Defaults to `[1, 1]`. If `dilations` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param dimRoundingMode The rounding mode used when computing output
 *     dimensions if pad is a number. If none is provided, it will not round
 *     and error if the output is of fractional size.
 */
/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function conv2d_<T extends Tensor3D|Tensor4D>(
    x: T|TensorLike, filter: Tensor4D|TensorLike,
    strides: [number, number]|number,
    pad: 'valid'|'same'|number|conv_util.ExplicitPadding,
    dataFormat: 'NHWC'|'NCHW' = 'NHWC',
    dilations: [number, number]|number = [1, 1],
    dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  const $x = convertToTensor(x, 'x', 'conv2d');
  const $filter = convertToTensor(filter, 'filter', 'conv2d');

  let x4D = $x as Tensor4D;
  let reshapedTo4D = false;

  if ($x.rank === 3) {
    reshapedTo4D = true;
    x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
  }

  util.assert(
      x4D.rank === 4,
      () => `Error in conv2d: input must be rank 4, but got rank ${x4D.rank}.`);
  util.assert(
      $filter.rank === 4,
      () => `Error in conv2d: filter must be rank 4, but got rank ` +
          `${$filter.rank}.`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in conv2d: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const inDepth = dataFormat === 'NHWC' ? x4D.shape[3] : x4D.shape[1];
  util.assert(
      inDepth === $filter.shape[2],
      () => `Error in conv2d: depth of input (${inDepth}) must match ` +
          `input depth for filter ${$filter.shape[2]}.`);
  util.assert(
      conv_util.eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in conv2D: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    const $dataFormat = conv_util.convertConv2DDataFormat(dataFormat);
    const convInfo = conv_util.computeConv2DInfo(
        x4D.shape, $filter.shape, strides, dilations, pad, dimRoundingMode,
        false, $dataFormat);
    const res = backend.conv2d(x4D, $filter, convInfo);

    save([x4D, $filter]);

    return res;
  };

  const inputs: Conv2DInputs = {x: x4D, filter: $filter};
  const attrs:
      Conv2DAttrs = {strides, pad, dataFormat, dilations, dimRoundingMode};

  const res = ENGINE.runKernelFunc(
      forward, inputs as {} as NamedTensorMap, null /* grad */, Conv2D,
      attrs as {} as NamedAttrMap);

  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as T;
  }
  return res as T;
}

export const conv2d = op({conv2d_});
