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
import {backend_util, KernelConfig, KernelFunc, MaxPool3D, MaxPool3DAttrs, MaxPool3DInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {Pool3DProgram} from '../pool_gpu';

export function maxPool3d(args: {
  inputs: MaxPool3DInputs,
  backend: MathBackendWebGL,
  attrs: MaxPool3DAttrs
}): TensorInfo {
  const {inputs, backend, attrs} = args;
  const {x} = inputs;
  const {filterSize, strides, pad, dataFormat, dimRoundingMode} = attrs;
  const dilations: [number, number, number] = [1, 1, 1];

  const convInfo = backend_util.computePool3DInfo(
      x.shape as [number, number, number, number, number], filterSize, strides,
      dilations, pad, dimRoundingMode, dataFormat);
  const maxPoolProgram = new Pool3DProgram(convInfo, 'max', false);
  return backend.runWebGLProgram(maxPoolProgram, [x], x.dtype);
}

export const maxPool3DConfig: KernelConfig = {
  kernelName: MaxPool3D,
  backendName: 'webgl',
  kernelFunc: maxPool3d as {} as KernelFunc
};
