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

import {backend_util, Dilation2DAttrs, Dilation2DBackpropFilter, Dilation2DBackpropFilterInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {Dilation2DBackpropFilterProgram} from '../dilation_backprop_gpu';

export function dilation2DBackpropFilter(args: {
  inputs: Dilation2DBackpropFilterInputs,
  attrs: Dilation2DAttrs,
  backend: MathBackendWebGL
}) {
  const {inputs, backend, attrs} = args;
  const {x, filter, dy} = inputs;
  const {strides, pad, dilations} = attrs;

  const convInfo = backend_util.computeDilation2DInfo(
      x.shape as [number, number, number, number],
      filter.shape as [number, number, number], strides, pad,
      'NHWC' /* dataFormat */, dilations);

  const program = new Dilation2DBackpropFilterProgram(convInfo);
  return backend.runWebGLProgram(program, [x, filter, dy], 'float32');
}

export const dilation2DBackpropFilterConfig: KernelConfig = {
  kernelName: Dilation2DBackpropFilter,
  backendName: 'webgl',
  kernelFunc: dilation2DBackpropFilter as {} as KernelFunc,
};
