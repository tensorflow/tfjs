/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {backend_util, DepthwiseConv2dNative, DepthwiseConv2dNativeAttrs, DepthwiseConv2dNativeInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {DepthwiseConv2DNCHWSharedProgram} from '../depthwise_conv2d_nchw_shared_webgpu';
import {DepthwiseConv2DVec4Program} from '../depthwise_conv2d_vec4_webgpu';
import {DepthwiseConv2DProgram} from '../depthwise_conv2d_webgpu';

export function depthwiseConv2dNative(args: {
  inputs: DepthwiseConv2dNativeInputs,
  attrs: DepthwiseConv2dNativeAttrs,
  backend: WebGPUBackend
}) {
  const {inputs, backend, attrs} = args;
  const {x, filter} = inputs;
  const {strides, pad, dataFormat, dilations, dimRoundingMode} = attrs;
  const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  let $dilations = dilations;
  if ($dilations == null) {
    $dilations = [1, 1];
  }

  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number, number], strides, $dilations,
      pad, dimRoundingMode, true /* depthwise */, $dataFormat);
  const dimensions = [
    {type: 'int32', data: [convInfo.padInfo.top, convInfo.padInfo.left]},
    {type: 'int32', data: [convInfo.inHeight, convInfo.inWidth]},
  ];

  const isChannelsLast = convInfo.dataFormat === 'channelsLast';
  let program: DepthwiseConv2DProgram|DepthwiseConv2DVec4Program|
      DepthwiseConv2DNCHWSharedProgram;
  if (!isChannelsLast && convInfo.inHeight > 16 && convInfo.inWidth > 16 &&
      convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
      convInfo.dilationWidth === 1 && convInfo.dilationHeight === 1 &&
      convInfo.inChannels === convInfo.outChannels) {
    program = new DepthwiseConv2DNCHWSharedProgram(
        convInfo.outShape, convInfo.filterHeight, convInfo.filterWidth);
  } else if (
      isChannelsLast && convInfo.outHeight > 4 && convInfo.outWidth > 4 &&
      convInfo.strideWidth <= 2 &&
      convInfo.inChannels === convInfo.outChannels &&
      convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
      convInfo.inChannels % 4 === 0) {
    program = new DepthwiseConv2DVec4Program(convInfo);
    dimensions.push({type: 'int32', data: [program.virtualWidth]});
  } else {
    program = new DepthwiseConv2DProgram(convInfo);
    dimensions.push(
        {type: 'int32', data: [convInfo.filterHeight]},
        {type: 'int32', data: [convInfo.filterWidth]},
        {type: 'int32', data: [convInfo.strideHeight, convInfo.strideWidth]}, {
          type: 'int32',
          data: [convInfo.dilationHeight, convInfo.dilationWidth]
        });
  }

  return backend.runWebGPUProgram(program, [x, filter], x.dtype, dimensions);
}

export const depthwiseConv2dNativeConfig: KernelConfig = {
  kernelName: DepthwiseConv2dNative,
  backendName: 'webgpu',
  kernelFunc: depthwiseConv2dNative as unknown as KernelFunc,
};
