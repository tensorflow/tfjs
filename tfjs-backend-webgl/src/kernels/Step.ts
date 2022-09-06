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

import {KernelConfig, KernelFunc, Step, StepAttrs, TensorInfo, UnaryInputs} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {CHECK_NAN_SNIPPET, UnaryOpProgram} from '../unaryop_gpu';

export function step(
    {inputs, attrs, backend}:
        {inputs: UnaryInputs, attrs: StepAttrs, backend: MathBackendWebGL}):
    TensorInfo {
  const {x} = inputs;
  const opSnippet = CHECK_NAN_SNIPPET + `
    return x > 0.0 ? 1.0 : float(${attrs.alpha});
  `;

  const program = new UnaryOpProgram(x.shape, opSnippet);

  return backend.runWebGLProgram(program, [x], x.dtype);
}

export const stepConfig: KernelConfig = {
  kernelName: Step,
  backendName: 'webgl',
  kernelFunc: step as {} as KernelFunc,
};
