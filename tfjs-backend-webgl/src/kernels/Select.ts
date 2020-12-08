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

import {KernelConfig, KernelFunc, Select, SelectInputs, TensorInfo, upcastType} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {SelectProgram} from '../select_gpu';

export function select(args: {inputs: SelectInputs, backend: MathBackendWebGL}):
    TensorInfo {
  const {inputs, backend} = args;
  const {condition, t, e} = inputs;

  const program =
      new SelectProgram(condition.shape.length, t.shape, t.shape.length);
  return backend.runWebGLProgram(
      program, [condition, t, e], upcastType(t.dtype, e.dtype));
}

export const selectConfig: KernelConfig = {
  kernelName: Select,
  backendName: 'webgl',
  kernelFunc: select as {} as KernelFunc
};
