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
import {Conv3D, Conv3DAttrs, Conv3DInputs} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor4D, Tensor5D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import {convertToTensor} from '../tensor_util_env';
import {TensorLike} from '../types';
import * as util from '../util';

import * as conv_util from './conv_util';
import {eitherStridesOrDilationsAreOne} from './conv_util';
import {op} from './operation';
import {reshape} from './reshape';

/**
 * Computes a 3D convolution over the input x.
 *
 * @param x The input tensor, of rank 5 or rank 4, of shape
 *     `[batch, depth, height, width, channels]`. If rank 4,
 * batch of 1 is assumed.
 * @param filter The filter, rank 5, of shape
 *     `[filterDepth, filterHeight, filterWidth, inChannels, outChannels]`.
 *      inChannels must match between input and filter.
 * @param strides The strides of the convolution: `[strideDepth, strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm.
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_guides/python/nn#Convolution](
 *          https://www.tensorflow.org/api_guides/python/nn#Convolution)
 * @param dataFormat: An optional string from: "NDHWC", "NCDHW". Defaults to
 *     "NDHWC". Specify the data format of the input and output data. With the
 *     default format "NDHWC", the data is stored in the order of: [batch,
 *     depth, height, width, channels]. Only "NDHWC" is currently supported.
 * @param dilations The dilation rates: `[dilationDepth, dilationHeight,
 *     dilationWidth]` in which we sample input values across the height
 *     and width dimensions in atrous convolution. Defaults to `[1, 1, 1]`.
 *     If `dilations` is a single number, then
 *     `dilationDepth == dilationHeight == dilationWidth`. If it is greater
 *     than 1, then all values of `strides` must be 1.
 */

/** @doc {heading: 'Operations', subheading: 'Convolution'} */
function conv3d_<T extends Tensor4D|Tensor5D>(
    x: T|TensorLike, filter: Tensor5D|TensorLike,
    strides: [number, number, number]|number, pad: 'valid'|'same',
    dataFormat: 'NDHWC'|'NCDHW' = 'NDHWC',
    dilations: [number, number, number]|number = [1, 1, 1]): T {
  const $x = convertToTensor(x, 'x', 'conv3d');
  const $filter = convertToTensor(filter, 'filter', 'conv3d');

  let x5D = $x as Tensor5D;
  let reshapedTo5D = false;

  if ($x.rank === 4) {
    reshapedTo5D = true;
    x5D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2], $x.shape[3]]);
  }
  util.assert(
      x5D.rank === 5,
      () => `Error in conv3d: input must be rank 5, but got rank ${x5D.rank}.`);
  util.assert(
      $filter.rank === 5,
      () => `Error in conv3d: filter must be rank 5, but got rank ` +
          `${$filter.rank}.`);
  util.assert(
      x5D.shape[4] === $filter.shape[3],
      () => `Error in conv3d: depth of input (${x5D.shape[4]}) must match ` +
          `input depth for filter ${$filter.shape[3]}.`);
  util.assert(
      eitherStridesOrDilationsAreOne(strides, dilations),
      () => 'Error in conv3D: Either strides or dilations must be 1. ' +
          `Got strides ${strides} and dilations '${dilations}'`);
  util.assert(
      dataFormat === 'NDHWC',
      () => `Error in conv3d: got dataFormat of ${
          dataFormat} but only NDHWC is currently supported.`);

  const forward: ForwardFunc<Tensor> = (backend, save) => {
    const convInfo = conv_util.computeConv3DInfo(
        x5D.shape, $filter.shape, strides, dilations, pad);
    const res = backend.conv3d(x5D, $filter, convInfo);

    save([x5D, $filter]);

    return res;
  };

  const inputs: Conv3DInputs = {x: x5D, filter: $filter};

  const attrs: Conv3DAttrs = {strides, pad, dataFormat, dilations};

  const res = ENGINE.runKernelFunc(
      forward, inputs as {} as NamedTensorMap, null /* grad */, Conv3D,
      attrs as {} as NamedAttrMap);

  if (reshapedTo5D) {
    return reshape(
               res, [res.shape[1], res.shape[2], res.shape[3], res.shape[4]]) as
        T;
  }
  return res as T;
}

export const conv3d = op({conv3d_});
