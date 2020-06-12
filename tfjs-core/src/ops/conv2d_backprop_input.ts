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
import {Conv2DBackpropInput, Conv2DBackpropInputAttrs, Conv2DBackpropInputInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor3D, Tensor4D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import * as util from '../util';

import {reshape} from './array_ops';
import * as conv_util from './conv_util';
import {op} from './operation';

/**
 * Computes the derivative of the input of a 2D convolution.
 *
 * @param xShape The shape of the input: [batch, height, width, inDepth].
 * If length of 3, batch of 1 is assumed.
 * @param dy The derivative of the output, of rank 4 or rank 3 of shape
 *   `[batch, outHeight, outWidth, outDepth]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter, rank 4, of shape
 *     `[filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm used:
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels].
 * @param dimRoundingMode The rounding mode used when computing output
 *     dimensions if pad is a number. If none is provided, it will not round
 *     and error if the output is of fractional size.
 */
function conv2DBackpropInput_<T extends Tensor3D|Tensor4D>(
    xShape: [number, number, number, number]|[number, number, number], dy: T,
    filter: Tensor4D, strides: [number, number]|number,
    pad: 'valid'|'same'|number|conv_util.ExplicitPadding,
    dataFormat: 'NHWC'|'NCHW' = 'NHWC',
    dimRoundingMode?: 'floor'|'round'|'ceil'): T {
  util.assert(
      xShape.length === dy.rank,
      () => `Length of inShape ` +
          `(${xShape.length}) and rank of dy (${dy.rank}) must match`);

  let xShape4D = xShape as [number, number, number, number];
  let dy4D = dy as Tensor4D;
  let reshapedTo4D = false;
  if (dy.rank === 3) {
    reshapedTo4D = true;
    dy4D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2]]);
    xShape4D = [1, xShape[0], xShape[1], xShape[2]];
  }

  util.assert(
      xShape4D.length === 4,
      () =>
          `Error in conv2dDerInput: inShape must be length 4, but got length ` +
          `${xShape4D.length}.`);
  util.assert(
      dy4D.rank === 4,
      () => `Error in conv2dDerInput: dy must be rank 4, but got ` +
          `rank ${dy4D.rank}`);
  util.assert(
      filter.rank === 4,
      () => `Error in conv2dDerInput: filter must be rank 4, but got ` +
          `rank ${filter.rank}`);
  const inDepth = dataFormat === 'NHWC' ? xShape4D[3] : xShape4D[1];
  const outDepth = dataFormat === 'NHWC' ? dy4D.shape[3] : dy4D.shape[1];
  util.assert(
      inDepth === filter.shape[2],
      () => `Error in conv2dDerInput: depth of input (${inDepth}) must ` +
          `match input depth for filter ${filter.shape[2]}.`);
  util.assert(
      outDepth === filter.shape[3],
      () => `Error in conv2dDerInput: depth of output (${outDepth}) must ` +
          `match output depth for filter ${filter.shape[3]}.`);
  if (dimRoundingMode != null) {
    util.assert(
        util.isInt(pad as number),
        () => `Error in conv2dDerInput: pad must be an integer when using, ` +
            `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
  }

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    const dilations = 1;

    const $dataFormat = conv_util.convertConv2DDataFormat(dataFormat);
    const convInfo = conv_util.computeConv2DInfo(
        xShape4D, filter.shape, strides, dilations, pad, dimRoundingMode, false,
        $dataFormat);

    const res = backend.conv2dDerInput(dy4D, filter, convInfo);

    save([dy4D, filter]);

    return res;
  };

  const inputs: Conv2DBackpropInputInputs = {dy: dy4D, filter};

  const attrs:
      Conv2DBackpropInputAttrs = {strides, pad, dataFormat, dimRoundingMode};

  const res = ENGINE.runKernelFunc(
      forward, inputs as {} as NamedTensorMap, null /* grad */,
      Conv2DBackpropInput, attrs as {} as NamedAttrMap);

  if (reshapedTo4D) {
    return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]) as T;
  }
  return res as T;
}

export const conv2DBackpropInput = op({conv2DBackpropInput_});
