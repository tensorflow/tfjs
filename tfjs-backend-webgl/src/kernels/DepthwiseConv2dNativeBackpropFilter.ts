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

import {backend_util, DepthwiseConv2dNativeBackpropFilter, DepthwiseConv2dNativeBackpropFilterAttrs, DepthwiseConv2dNativeBackpropFilterInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {DepthwiseConv2DDerFilterProgram} from '../conv_backprop_gpu_depthwise';

export function depthwiseConv2dNativeBackpropFilter(args: {
  inputs: DepthwiseConv2dNativeBackpropFilterInputs,
  attrs: DepthwiseConv2dNativeBackpropFilterAttrs,
  backend: MathBackendWebGL
}) {
  const {inputs, backend, attrs} = args;
  const {x, dy} = inputs;
  const {strides, dilations, pad, dimRoundingMode, filterShape} = attrs;

  const convInfo = backend_util.computeConv2DInfo(
      x.shape as [number, number, number, number], filterShape, strides,
      dilations, pad, dimRoundingMode, true /* depthwise */);

  const program = new DepthwiseConv2DDerFilterProgram(convInfo);
  return backend.runWebGLProgram(program, [x, dy], 'float32');
}

export const depthwiseConv2dNativeBackpropFilterConfig: KernelConfig = {
  kernelName: DepthwiseConv2dNativeBackpropFilter,
  backendName: 'webgl',
  kernelFunc: depthwiseConv2dNativeBackpropFilter as {} as KernelFunc
};
