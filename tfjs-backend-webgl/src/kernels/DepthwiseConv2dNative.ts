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

import {backend_util, DepthwiseConv2dNative, DepthwiseConv2dNativeAttrs, DepthwiseConv2dNativeInputs, env, KernelConfig, KernelFunc, util} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {DepthwiseConv2DProgram} from '../conv_gpu_depthwise';
import {DepthwiseConvPacked2DProgram} from '../conv_packed_gpu_depthwise';

export function depthwiseConv2dNative(args: {
  inputs: DepthwiseConv2dNativeInputs,
  attrs: DepthwiseConv2dNativeAttrs,
  backend: MathBackendWebGL
}) {
  const {inputs, backend, attrs} = args;
  const {x, filter} = inputs;
  const {strides, pad, dilations, dimRoundingMode} = attrs;

  let $dilations = dilations;
  if ($dilations == null) {
    $dilations = [1, 1];
  }

  util.assert(
      backend_util.eitherStridesOrDilationsAreOne(strides, $dilations),
      () => 'Error in depthwiseConv2d: Either strides or dilations must be ' +
          `1. Got strides ${strides} and dilations '${$dilations}'`);

  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number, number], strides, $dilations,
      pad, dimRoundingMode, true /* depthwise */);

  let program: DepthwiseConv2DProgram|DepthwiseConvPacked2DProgram;
  if (env().getBool('WEBGL_PACK_DEPTHWISECONV') && convInfo.strideWidth <= 2 &&
      convInfo.outChannels / convInfo.inChannels === 1) {
    program = new DepthwiseConvPacked2DProgram(convInfo);
  } else {
    program = new DepthwiseConv2DProgram(convInfo);
  }

  return backend.runWebGLProgram(program, [x, filter], 'float32');
}

export const depthwiseConv2dNativeConfig: KernelConfig = {
  kernelName: DepthwiseConv2dNative,
  backendName: 'webgl',
  kernelFunc: depthwiseConv2dNative as {} as KernelFunc,
};
