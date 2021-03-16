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

import {backend_util, Conv2D, Conv2DAttrs, Conv2DInputs, env, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {conv2dByMatMul, conv2dWithIm2Col} from './Conv2D_impl';
import {Conv2DMMVec4Program} from './conv2d_mm_vec4_webgpu';
import {Conv2DMMProgram} from './conv2d_mm_webgpu';
import {Conv2DNaiveProgram} from './conv2d_naive_webgpu';

export function conv2d(
    args: {inputs: Conv2DInputs, attrs: Conv2DAttrs, backend: WebGPUBackend}) {
  const {inputs, attrs, backend} = args;
  const {x, filter} = inputs;
  const {strides, pad, dataFormat, dilations, dimRoundingMode} = attrs;
  const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number, number], strides, dilations, pad,
      dimRoundingMode, false /* depthwise */, $dataFormat);
  if (convInfo.filterHeight === 1 && convInfo.filterWidth === 1 &&
      convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1 &&
      convInfo.strideHeight === 1 && convInfo.strideWidth === 1 &&
      (convInfo.padInfo.type === 'SAME' || convInfo.padInfo.type === 'VALID')) {
    return conv2dByMatMul({x, filter, convInfo, backend});
  }

  if (env().getBool('WEBGPU_CONV_SEPARATE_IM2COL_SHADER') && x.shape[0] === 1) {
    return conv2dWithIm2Col({x, filter, convInfo, backend});
  }

  let program: Conv2DMMProgram|Conv2DNaiveProgram|Conv2DMMVec4Program;

  if (env().getBool('WEBGPU_USE_NAIVE_CONV2D')) {
    // TODO(kainino0x): This may be obsolete, but is kept for reference.
    program = new Conv2DNaiveProgram(convInfo);
  } else if (
      // TODO(jiajia.qin@intel.com): It seems that the vec4 version is not
      // good if convInfo.outChannels is too small. For example, input = [1,
      // 128, 128, 4], filter = [25, 25, 4, 4]. In this case, lots of theads
      // will run idle. So temporarily, use 64 as the threshold.
      (convInfo.inChannels % 4 === 0 ||
       (convInfo.inChannels === 3 && convInfo.padInfo.type === 'VALID')) &&
      convInfo.outChannels % 4 === 0 && convInfo.outChannels >= 64) {
    program = new Conv2DMMVec4Program(convInfo);
  } else {
    program = new Conv2DMMProgram(convInfo);
  }

  const padInfo = [convInfo.padInfo.top, convInfo.padInfo.left];

  const dimensions = [
    convInfo.filterHeight, convInfo.filterWidth, ...padInfo,
    convInfo.strideHeight, convInfo.strideWidth, convInfo.dilationHeight,
    convInfo.dilationWidth
  ];
  const uniformData = new Int32Array(dimensions);

  return backend.runWebGPUProgram(program, [x, filter], x.dtype, uniformData);
}

export const conv2DConfig: KernelConfig = {
  kernelName: Conv2D,
  backendName: 'webgpu',
  kernelFunc: conv2d as {} as KernelFunc
};
