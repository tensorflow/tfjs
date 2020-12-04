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

import {env, KernelConfig, KernelFunc, Neg, NegInputs, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';
import {MathBackendWebGL} from '../backend_webgl';
import {negImplCPU} from '../kernel_utils/shared';
import {UnaryOpProgram} from '../unaryop_gpu';
import {UnaryOpPackedProgram} from '../unaryop_packed_gpu';

const NEG = `return -x;`;

// This doesn't use unaryKernelFunc because negImplCPU is not of type
// SimpleUnaryKernelImplCPU.
export function neg(args: {inputs: NegInputs, backend: MathBackendWebGL}):
    TensorInfo {
  const {inputs, backend} = args;
  const {x} = inputs;

  if (backend.shouldExecuteOnCPU([x])) {
    const xData = backend.texData.get(x.dataId);
    const [outValues, newShape] =
        negImplCPU(xData.values as TypedArray, x.shape, x.dtype);
    return backend.makeTensorInfo(newShape, x.dtype, outValues);
  }

  let program: UnaryOpProgram|UnaryOpPackedProgram;
  if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
    program = new UnaryOpPackedProgram(x.shape, NEG);
  } else {
    program = new UnaryOpProgram(x.shape, NEG);
  }

  return backend.runWebGLProgram(program, [x], x.dtype);
}

export const negConfig: KernelConfig = {
  kernelName: Neg,
  backendName: 'webgl',
  kernelFunc: neg as {} as KernelFunc
};
