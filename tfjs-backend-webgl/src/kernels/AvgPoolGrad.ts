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
import {AvgPoolGrad, AvgPoolGradAttrs, AvgPoolGradInputs, backend_util, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {AvgPool2DBackpropProgram} from '../avg_pool_backprop_gpu';
import {MathBackendWebGL} from '../backend_webgl';
import {assertNotComplex} from '../webgl_util';

export function avgPoolGrad(args: {
  inputs: AvgPoolGradInputs,
  backend: MathBackendWebGL,
  attrs: AvgPoolGradAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, input} = inputs;
  const x = input;
  assertNotComplex([dy, input], 'avgPoolGrad');
  const {filterSize, strides, pad} = attrs;

  const convInfo = backend_util.computePool2DInfo(
      x.shape as [number, number, number, number], filterSize, strides,
      1 /* dilations */, pad);
  const avgPoolBackpropProgram = new AvgPool2DBackpropProgram(convInfo);
  return backend.runWebGLProgram(avgPoolBackpropProgram, [dy], x.dtype);
}

export const avgPoolGradConfig: KernelConfig = {
  kernelName: AvgPoolGrad,
  backendName: 'webgl',
  kernelFunc: avgPoolGrad as {} as KernelFunc
};
