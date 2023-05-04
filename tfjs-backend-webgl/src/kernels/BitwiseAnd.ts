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

import {BitwiseAnd, BitwiseAndInputs, env, KernelConfig, KernelFunc, TensorInfo, TypedArray} from '@tensorflow/tfjs-core';
import {MathBackendWebGL} from '../backend_webgl';
import {BinaryOpProgram} from '../binaryop_gpu';
import {BinaryOpPackedProgram} from '../binaryop_packed_gpu';
import {bitwiseAndImplCPU as cpuBitwiseAnd} from '../kernel_utils/shared';

export const BITWISEAND = `
  int r = int(a.r) & int(b.r);
  int g = int(a.g) & int(b.g);
  int rb = int(a.b) & int(b.b);
  int ra = int(a.a) & int(b.a);
  return vec4(r, g, rb, ra);
`;

export const BITWISEAND_UNPACKED = `
  return float(int(a.r) & int(b.r));
`;

export function bitwiseAnd(args: {
  inputs: BitwiseAndInputs,
  backend: MathBackendWebGL,
}): TensorInfo {
  const {inputs, backend} = args;
  const {a, b} = inputs;
  const shouldUsePackedProgram = env().getBool('WEBGL_PACK_BINARY_OPERATIONS');
  const versionNumber = env().getNumber('WEBGL_VERSION');

  // The type of a and b are ensured to be `int32` in core, therefore no need to
  // consider other type situations.
  if ((backend.shouldExecuteOnCPU([a, b])) || versionNumber === 1) {
    const aVals = backend.texData.get(a.dataId).values as TypedArray;
    const bVals = backend.texData.get(b.dataId).values as TypedArray;
    const [outValues, outShape] =
        cpuBitwiseAnd(a.shape, b.shape, aVals, bVals, a.dtype);

    const out = backend.makeTensorInfo(outShape, a.dtype);
    const outData = backend.texData.get(out.dataId);
    outData.values = outValues;
    return out;
  }

  let program: BinaryOpProgram|BinaryOpPackedProgram;
  if (shouldUsePackedProgram) {
    program = new BinaryOpPackedProgram(BITWISEAND, a.shape, b.shape, false);
  } else {
    program = new BinaryOpProgram(BITWISEAND_UNPACKED, a.shape, b.shape);
  }

  return backend.runWebGLProgram(program, [a, b], a.dtype);
}

export const bitwiseAndConfig: KernelConfig = {
  kernelName: BitwiseAnd,
  backendName: 'webgl',
  kernelFunc: bitwiseAnd as unknown as KernelFunc
};
