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

import {EluGrad, EluGradInputs, env, KernelConfig, KernelFunc, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {BinaryOpProgram} from '../binaryop_gpu';
import {BinaryOpPackedProgram} from '../binaryop_packed_gpu';

const ELU_DER = `return (b >= 1.0) ? a : a * (b + 1.0);`;
const ELU_DER_PACKED = `
  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));
  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));
`;

export const eluGrad =
    (args: {inputs: EluGradInputs, backend: MathBackendWebGL}): TensorInfo => {
      const {inputs, backend} = args;
      const {dy, y} = inputs;

      const program = env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
          new BinaryOpPackedProgram(ELU_DER_PACKED, dy.shape, y.shape) :
          new BinaryOpProgram(ELU_DER, dy.shape, y.shape);
      return backend.runWebGLProgram(program, [dy, y], dy.dtype);
    };

export const eluGradConfig: KernelConfig = {
  kernelName: EluGrad,
  backendName: 'webgl',
  kernelFunc: eluGrad as {} as KernelFunc
};
