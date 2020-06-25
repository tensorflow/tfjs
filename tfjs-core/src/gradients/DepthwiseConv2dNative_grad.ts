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
import {DepthwiseConv2dNative, DepthwiseConv2dNativeAttrs} from '../kernel_names';
import {GradConfig, NamedAttrMap} from '../kernel_registry';
import * as conv_util from '../ops/conv_util';
import {depthwiseConv2dNativeBackpropFilter} from '../ops/depthwise_conv2d_native_backprop_filter';
import {depthwiseConv2dNativeBackpropInput} from '../ops/depthwise_conv2d_native_backprop_input';
import {Tensor, Tensor4D} from '../tensor';
import * as util from '../util';

export const depthwiseConv2dNativeGradConfig: GradConfig = {
  kernelName: DepthwiseConv2dNative,
  inputsToSave: ['x', 'filter'],
  gradFunc: (dy: Tensor4D, saved: Tensor[], attrs: NamedAttrMap) => {
    const {dilations, strides, pad, dimRoundingMode} =
        attrs as {} as DepthwiseConv2dNativeAttrs;

    const $dilations = dilations == null ? [1, 1] : dilations;

    util.assert(
        conv_util.tupleValuesAreOne($dilations),
        () => 'Error in gradient of depthwiseConv2dNative: dilation rates ' +
            `greater than 1 are not yet supported. Got dilations ` +
            `'${$dilations}'`);

    const [x, filter] = saved as [Tensor4D, Tensor4D];

    util.assert(
        x.rank === 4,
        () => `Error in gradient of depthwiseConv2dNative: input must be ` +
            `rank 4, but got rank ${x.rank}.`);
    util.assert(
        filter.rank === 4,
        () => `Error in gradient of depthwiseConv2dNative: filter must be ` +
            `rank 4, but got rank ${filter.rank}.`);
    util.assert(
        x.shape[3] === filter.shape[2],
        () => `Error in gradient of depthwiseConv2d: number of input ` +
            `channels (${x.shape[3]}) must match the inChannels dimension ` +
            `in filter ${filter.shape[2]}.`);

    util.assert(
        conv_util.eitherStridesOrDilationsAreOne(strides, $dilations),
        () => 'Error in gradient of depthwiseConv2d: Either strides or ' +
            `dilations must be  1. Got strides ${strides} and dilations ` +
            `'${$dilations}'.`);

    if (dimRoundingMode != null) {
      util.assert(
          util.isInt(pad as number),
          () =>
              `Error in depthwiseConv2d: pad must be an integer when using, ` +
              `dimRoundingMode ${dimRoundingMode} but got pad ${pad}.`);
    }

    const convInfo = conv_util.computeConv2DInfo(
        x.shape, filter.shape, strides, $dilations as number | [number, number],
        pad, dimRoundingMode, true /* depthwise */);

    return {
      x: () =>
          depthwiseConv2dNativeBackpropInput(x.shape, dy, filter, convInfo),
      filter: () =>
          depthwiseConv2dNativeBackpropFilter(x, dy, filter.shape, convInfo),
    };
  }
};
