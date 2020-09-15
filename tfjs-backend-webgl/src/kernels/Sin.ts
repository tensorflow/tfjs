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

import {KernelConfig, KernelFunc, Sin, SinInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {SIN, UnaryOpProgram} from '../unaryop_gpu';

export const sinKernelFunc:
    (params: {inputs: SinInputs, backend: MathBackendWebGL}) =>
        TensorInfo | TensorInfo[] = ({inputs, backend}) => {
          const {x} = inputs;
          const program = new UnaryOpProgram(x.shape, SIN);
          return backend.runWebGLProgram(program, [x], x.dtype);
        };

export const sinConfig: KernelConfig = {
  kernelName: Sin,
  backendName: 'webgl',
  kernelFunc: sinKernelFunc as {} as KernelFunc,
};
