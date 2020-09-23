
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

import {BinaryInputs, env, KernelConfig, Multiply, TensorInfo} from '@tensorflow/tfjs-core';

// import {binaryKernelFunc} from '../kernel_utils/kernel_funcs_utils';
import {MathBackendWebGL} from '../backend_webgl';
import {BinaryOpProgram} from '../binaryop_gpu';
import {BinaryOpPackedProgram} from '../binaryop_packed_gpu';
// import * as binaryop_complex_gpu from './binaryop_complex_gpu';
// import {BinaryOpComplexProgram} from './binaryop_complex_gpu';

const MUL = 'return a * b;';

export function multiply(
    args: {inputs: BinaryInputs, backend: MathBackendWebGL}): TensorInfo {
  const {inputs, backend} = args;
  const {a, b} = inputs;

  if (a.dtype === 'complex64') {
  }

  // const shouldExecuteOnCPU = backend.shouldExecuteOnCPU([a, b]);
  // if (shouldExecuteOnCPU) {
  //   return
  // }

  let program: BinaryOpProgram|BinaryOpPackedProgram;
  if (env().getBool('WEBGL_PACK_BINARY_OPERATIONS')) {
    program = new BinaryOpPackedProgram(MUL, a.shape, b.shape);
  } else {
    program = new BinaryOpProgram(MUL, a.shape, b.shape);
  }

  return backend.runWebGLProgram(program, [a, b], a.dtype);
}

export const multiplyConfig: KernelConfig = {
  kernelName: Multiply,
  backendName: 'webgl',
  kernelFunc: multiply
};
