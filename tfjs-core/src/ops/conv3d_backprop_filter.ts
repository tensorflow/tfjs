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
import {Conv3DBackpropFilterAttrs, Conv3DBackpropFilterInputs, Conv3DBackpropFilterV2} from '../kernel_names';
import {NamedAttrMap} from '../kernel_registry';
import {Tensor, Tensor4D, Tensor5D} from '../tensor';
import {NamedTensorMap} from '../tensor_types';
import * as util from '../util';

import * as conv_util from './conv_util';
import {op} from './operation';
import {reshape} from './reshape';

/**
 * Computes the derivative of the filter of a 3D convolution.
 *
 * @param x The input tensor, of rank 5 or rank 4 of shape
 *     [batch, depth, height, width, inChannels]. If rank 4, batch of 1 is
 *     assumed.
 * @param dy The dy image, of rank 5 or rank 4, of shape
 *     [batch, depth, height, width, outDepth]. If rank 4, batch of 1 is
 *     assumed.
 * @param filterShape The shape of the filter, length 5,
 *     [filterDepth, filterHeight, filterWidth, inDepth, outDepth].
 * @param strides The strides of the convolution: [strideDepth, strideHeight,
 * strideWidth].
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 */
function conv3DBackpropFilter_<T extends Tensor4D|Tensor5D>(
    x: T, dy: T, filterShape: [number, number, number, number, number],
    strides: [number, number, number]|number, pad: 'valid'|'same'): Tensor5D {
  let x5D = x as Tensor5D;
  if (x.rank === 4) {
    x5D = reshape(x, [1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]]);
  }
  let dy5D = dy as Tensor5D;
  if (dy5D.rank === 4) {
    dy5D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2], dy.shape[3]]);
  }
  util.assert(
      x5D.rank === 5,
      () => `Error in conv3dDerFilter: input must be rank 5, but got shape ` +
          `${x5D.shape}.`);
  util.assert(
      dy5D.rank === 5,
      () => `Error in conv3dDerFilter: dy must be rank 5, but got shape ` +
          `${dy5D.shape}.`);
  util.assert(
      filterShape.length === 5,
      () => `Error in conv3dDerFilter: filterShape must be length 5, but got ` +
          `${filterShape}.`);
  util.assert(
      x5D.shape[4] === filterShape[3],
      () => `Error in conv3dDerFilter: depth of input ${x5D.shape[4]}) must ` +
          `match input depth in filter (${filterShape[3]}.`);
  util.assert(
      dy5D.shape[4] === filterShape[4],
      () => `Error in conv3dDerFilter: depth of dy (${dy5D.shape[4]}) must ` +
          `match output depth for filter (${filterShape[4]}).`);

  const forward: ForwardFunc<Tensor> = backend => {
    const dilations = 1;

    const convInfo = conv_util.computeConv3DInfo(
        x5D.shape, filterShape, strides, dilations, pad);

    return backend.conv3dDerFilter(x5D, dy5D, convInfo);
  };

  const inputs: Conv3DBackpropFilterInputs = {x: x5D, y: dy5D};

  const attrs: Conv3DBackpropFilterAttrs = {strides, pad};

  return ENGINE.runKernelFunc(
             forward, inputs as {} as NamedTensorMap, null,
             Conv3DBackpropFilterV2, attrs as {} as NamedAttrMap) as Tensor5D;
}

export const conv3DBackpropFilter = op({conv3DBackpropFilter_});
