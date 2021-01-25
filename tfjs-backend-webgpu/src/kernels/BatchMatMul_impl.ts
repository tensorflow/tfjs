/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {backend_util, env, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {MatMulPackedProgram} from './matmul_packed_webgpu';
import {MatMulPackedVec4Program} from './matmul_packed_vec4_webgpu';

type BatchMatMulConfig = {
  a: TensorInfo,
  b: TensorInfo,
  transposeA: boolean,
  transposeB: boolean,
  backend: WebGPUBackend,
  bias?: TensorInfo,
  preluActivationWeights?: TensorInfo,
  leakyreluAlpha?: number,
  activation?: backend_util.Activation
};

export function batchMatMulImpl({
  a,
  b,
  transposeA,
  transposeB,
  backend,
  bias = null,
  preluActivationWeights = null,
  leakyreluAlpha = 0,
  activation = null
}: BatchMatMulConfig): TensorInfo {
  const outerShapeA = transposeA ? a.shape[2] : a.shape[1];
  const outerShapeB = transposeB ? b.shape[1] : b.shape[2];
  const [batch, , ] = a.shape;

  const hasBias = bias != null;
  const hasPreluActivationWeights = preluActivationWeights != null;
  const useVec4 = a.shape[2] % 4 === 0 && b.shape[2] % 4 === 0 &&
      !transposeA && !transposeB;
  const fusedActivation = activation ?
      backend.mapActivationToShaderProgram(activation, useVec4) :
      null;
  let program: MatMulPackedProgram|MatMulPackedVec4Program;
  if (useVec4) {
    // TODO: Currently we need to make sure that a.shape[2] and b.shape[2]
    // are divisible by 4 since we use vec4 to get data. In future, we can
    // remove this limitation by insert 0 to pack data.
    program = new MatMulPackedVec4Program(
        a.shape as [number, number, number], [batch, outerShapeA, outerShapeB],
        env().get('WEBGPU_MATMUL_WORK_PER_THREAD') as number, hasBias,
        fusedActivation, hasPreluActivationWeights);
  } else {
    program = new MatMulPackedProgram(
        a.shape as [number, number, number], [batch, outerShapeA, outerShapeB],
        env().get('WEBGPU_MATMUL_WORK_PER_THREAD') as number, transposeA,
        transposeB, hasBias, fusedActivation, hasPreluActivationWeights);
  }
  const inputs: TensorInfo[] = [a, b];
  if (bias) {
    inputs.push(bias);
  }
  if (preluActivationWeights) {
    inputs.push(preluActivationWeights);
  }
  return backend.runWebGPUProgram(program, inputs, a.dtype);
}
