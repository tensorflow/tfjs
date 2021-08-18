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

import {backend_util, Conv3D, Conv3DAttrs, Conv3DInputs, KernelConfig, KernelFunc} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {Conv3DProgram} from '../conv_gpu';

export function conv3D(
    args:
        {inputs: Conv3DInputs, attrs: Conv3DAttrs, backend: MathBackendWebGL}) {
  const {inputs, backend, attrs} = args;
  const {x, filter} = inputs;
  const {strides, pad, dilations} = attrs;

  const convInfo = backend_util.computeConv3DInfo(
      x.shape as [number, number, number, number, number],
      filter.shape as [number, number, number, number, number], strides,
      dilations, pad);

  const program = new Conv3DProgram(convInfo);
  return backend.runWebGLProgram(program, [x, filter], 'float32');
}

export const conv3DConfig: KernelConfig = {
  kernelName: Conv3D,
  backendName: 'webgl',
  kernelFunc: conv3D as {} as KernelFunc,
};
