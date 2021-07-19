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
import {AvgPool3DGrad, AvgPool3DGradAttrs, AvgPool3DGradInputs, backend_util, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {AvgPool3DBackpropProgram} from '../avg_pool_backprop_gpu';
import {MathBackendWebGL} from '../backend_webgl';

export function avgPool3DGrad(args: {
  inputs: AvgPool3DGradInputs,
  backend: MathBackendWebGL,
  attrs: AvgPool3DGradAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {dy, input} = inputs;
  const x = input;
  const {filterSize, strides, pad, dimRoundingMode} = attrs;
  const dilations: [number, number, number] = [1, 1, 1];

  const convInfo = backend_util.computePool3DInfo(
      x.shape as [number, number, number, number, number], filterSize, strides,
      dilations, pad, dimRoundingMode);
  const avgPoolBackpropProgram = new AvgPool3DBackpropProgram(convInfo);
  return backend.runWebGLProgram(avgPoolBackpropProgram, [dy], x.dtype);
}

export const avgPoolGrad3DConfig: KernelConfig = {
  kernelName: AvgPool3DGrad,
  backendName: 'webgl',
  kernelFunc: avgPool3DGrad as {} as KernelFunc
};
