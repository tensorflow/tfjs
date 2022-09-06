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

import {backend_util, Conv2DBackpropInput, Conv2DBackpropInputAttrs, Conv2DBackpropInputInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {Conv2DDerInputProgram} from '../conv_backprop_gpu';

export function conv2DBackpropInput(args: {
  inputs: Conv2DBackpropInputInputs,
  attrs: Conv2DBackpropInputAttrs,
  backend: MathBackendWebGL
}) {
  const {inputs, backend, attrs} = args;
  const {dy, filter} = inputs;
  const {inputShape, strides, pad, dataFormat, dimRoundingMode} = attrs;

  const $dataFormat = backend_util.convertConv2DDataFormat(dataFormat);
  const convInfo = backend_util.computeConv2DInfo(
      inputShape, filter.shape as [number, number, number, number], strides,
      1 /* dilations */, pad, dimRoundingMode, false, $dataFormat);

  const program = new Conv2DDerInputProgram(convInfo);
  return backend.runWebGLProgram(program, [dy, filter], 'float32');
}

export const conv2DBackpropInputConfig: KernelConfig = {
  kernelName: Conv2DBackpropInput,
  backendName: 'webgl',
  kernelFunc: conv2DBackpropInput as {} as KernelFunc,
};
