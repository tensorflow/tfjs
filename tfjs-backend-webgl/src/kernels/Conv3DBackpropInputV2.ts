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

import {backend_util, Conv3DBackpropInputV2, Conv3DBackpropInputV2Attrs, Conv3DBackpropInputV2Inputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {Conv3DDerInputProgram} from '../conv_backprop_gpu';

export function conv3DBackpropInput(args: {
  inputs: Conv3DBackpropInputV2Inputs,
  attrs: Conv3DBackpropInputV2Attrs,
  backend: MathBackendWebGL
}) {
  const {inputs, backend, attrs} = args;
  const {dy, filter} = inputs;
  const {pad, strides, inputShape} = attrs;

  const convInfo = backend_util.computeConv3DInfo(
      inputShape, filter.shape as [number, number, number, number, number],
      strides, 1 /* dilations */, pad);

  const program = new Conv3DDerInputProgram(convInfo);
  return backend.runWebGLProgram(program, [dy, filter], 'float32');
}

export const conv3DBackpropInputConfig: KernelConfig = {
  kernelName: Conv3DBackpropInputV2,
  backendName: 'webgl',
  kernelFunc: conv3DBackpropInput as {} as KernelFunc,
};
