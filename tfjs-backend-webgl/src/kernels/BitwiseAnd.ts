/**
 * @license
 * Copyright 2023 Google LLC.
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

import {BitwiseAnd, BitwiseAndInputs, DataTypeMap, env, KernelConfig, KernelFunc, TensorInfo, upcastType} from '@tensorflow/tfjs-core';
import {MathBackendWebGL} from '../backend_webgl';
import {BinaryOpProgram} from '../binaryop_gpu';
import {BinaryOpPackedProgram} from '../binaryop_packed_gpu';

export const BITWISEAND = `
  vec4 result;
  int r = int(a.r) & int(b.r);
  int g = int(a.g) & int(b.g);
  int rb = int(a.b) & int(b.b);
  int ra = int(a.a) & int(b.a);
  return vec4(r, g, rb, ra);
`;

export function bitwiseAnd(args: {
  inputs: BitwiseAndInputs,
  backend: MathBackendWebGL,
  checkOutOfBounds: boolean,
  dtype?: keyof DataTypeMap
}): TensorInfo {
  const {inputs, backend, checkOutOfBounds, dtype} = args;
  const {a, b} = inputs;
  const webglBackend = backend as MathBackendWebGL;
  const $dtype = dtype || upcastType(a.dtype, b.dtype);
  const shouldUsePackedProgram =
      env().getBool('WEBGL_PACK_BINARY_OPERATIONS') && BITWISEAND != null;
  const versionNumber = env().getNumber('WEBGL_VERSION');
  if (versionNumber !== 2) {
    throw new Error(
        `Unsupported webgl version. Current webgl version: ${versionNumber}`);
  }
  let program: BinaryOpProgram|BinaryOpPackedProgram;
  if (shouldUsePackedProgram) {
    program = new BinaryOpPackedProgram(
        BITWISEAND, a.shape, b.shape, checkOutOfBounds);
  } else {
    program = new BinaryOpProgram(BITWISEAND, a.shape, b.shape);
  }

  return webglBackend.runWebGLProgram(program, [a, b], $dtype);
}

export const bitwiseAndConfig: KernelConfig = {
  kernelName: BitwiseAnd,
  backendName: 'webgl',
  kernelFunc: bitwiseAnd as unknown as KernelFunc
};
