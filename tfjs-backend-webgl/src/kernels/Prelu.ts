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

import {env, KernelConfig, KernelFunc, Prelu, PreluInputs, TensorInfo} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {BinaryOpProgram} from '../binaryop_gpu';
import {BinaryOpPackedProgram} from '../binaryop_packed_gpu';

export const PRELU = `return (a < 0.) ? b * a : a;`;
export const PRELU_PACKED = `
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;

export function prelu(args: {inputs: PreluInputs, backend: MathBackendWebGL}):
    TensorInfo {
  const {inputs, backend} = args;
  const {x, alpha} = inputs;

  const program = env().getBool('WEBGL_PACK_BINARY_OPERATIONS') ?
      new BinaryOpPackedProgram(PRELU_PACKED, x.shape, alpha.shape) :
      new BinaryOpProgram(PRELU, x.shape, alpha.shape);
  return backend.runWebGLProgram(program, [x, alpha], x.dtype);
}

export const preluConfig: KernelConfig = {
  kernelName: Prelu,
  backendName: 'webgl',
  kernelFunc: prelu as {} as KernelFunc
};
