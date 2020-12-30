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

import {Abs, AbsInputs, env, KernelConfig, KernelFunc, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';

import {MathBackendWebGL} from '../backend_webgl';
import {simpleAbsImplCPU} from '../kernel_utils/shared';
import {UnaryOpProgram} from '../unaryop_gpu';
import {UnaryOpPackedProgram} from '../unaryop_packed_gpu';

const ABS = `return abs(x);`;

export function abs(args: {inputs: AbsInputs, backend: MathBackendWebGL}):
    TensorInfo {
  const {inputs, backend} = args;
  const {x} = inputs;

  // TODO: handle cases when x is complex. Once the cpu implementation
  // can handle complex values, refactor to use unaryKernelFunc.
  if (backend.shouldExecuteOnCPU([x]) && x.dtype !== 'complex64') {
    const xData = backend.texData.get(x.dataId);
    const outValues = simpleAbsImplCPU(xData.values as TypedArray);
    return backend.makeTensorInfo(x.shape, x.dtype, outValues);
  }

  let program: UnaryOpProgram|UnaryOpPackedProgram;
  if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
    program = new UnaryOpPackedProgram(x.shape, ABS);
  } else {
    program = new UnaryOpProgram(x.shape, ABS);
  }
  return backend.runWebGLProgram(program, [x], x.dtype);
}

export const absConfig: KernelConfig = {
  kernelName: Abs,
  backendName: 'webgl',
  kernelFunc: abs as {} as KernelFunc
};
