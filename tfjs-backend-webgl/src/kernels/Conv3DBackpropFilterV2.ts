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

import {backend_util, Conv3DBackpropFilterV2, Conv3DBackpropFilterV2Attrs, Conv3DBackpropFilterV2Inputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {Conv3DDerFilterProgram} from '../conv_backprop_gpu';

export function conv3DBackpropFilterV2(args: {
  inputs: Conv3DBackpropFilterV2Inputs,
  attrs: Conv3DBackpropFilterV2Attrs,
  backend: MathBackendWebGL
}) {
  const {inputs, backend, attrs} = args;
  const {x, dy} = inputs;
  const {strides, pad, filterShape} = attrs;

  const convInfo = backend_util.computeConv3DInfo(
      x.shape as [number, number, number, number, number], filterShape, strides,
      1 /* dilations */, pad);

  const program = new Conv3DDerFilterProgram(convInfo);
  return backend.runWebGLProgram(program, [x, dy], 'float32');
}

export const conv3DBackpropFilterV2Config: KernelConfig = {
  kernelName: Conv3DBackpropFilterV2,
  backendName: 'webgl',
  kernelFunc: conv3DBackpropFilterV2 as {} as KernelFunc
};
