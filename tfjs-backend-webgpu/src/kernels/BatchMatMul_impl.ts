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

import {backend_util, engine, env, TensorInfo} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {MatMulProgram} from './matmul_webgpu';
import {MatMulPackedProgram} from './matmul_packed_webgpu';
import {MatMulPackedVec4Program} from './matmul_packed_vec4_webgpu';

// Empirically determined minimal shared dimension in matmul before we forward
// to a.mul(b).sum() in order to take advantage of GPU parallelism. See
// https://github.com/tensorflow/tfjs-core/pull/1379 for benchmarks.
export const MATMUL_SHARED_DIM_THRESHOLD = 1000;

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

    const dataId = backend.write(
        null /*values*/, [batch, outerShapeA, outerShapeB], a.dtype);
    const output = engine().makeTensorFromDataId(
        dataId, [batch, outerShapeA, outerShapeB], a.dtype, backend);

    //let program: MatMulProgram|MatMulPackedProgram|MatMulPackedVec4Program;
    let program: MatMulProgram;
    // TODO: We should eventually use the blocked version, but keeping around
    // the old version while we try to understand conditions under which blocked
    // is faster.
    if (env().get('WEBGPU_MATMUL_WORK_PER_THREAD') === 0) {
      program = new MatMulProgram(
          a.shape as [number, number, number],
          output.shape as [number, number, number],
          transposeA, transposeB);
    } else if (
        a.shape[2] % 4 === 0 && b.shape[2] % 4 === 0 && !transposeA &&
        !transposeB) {
      // TODO: Currently we need to make sure that a.shape[2] and b.shape[2] are
      // divisible by 4 since we use vec4 to get data. In future, we can remove
      // this limitation by insert 0 to pack data.
      program = new MatMulPackedVec4Program(
          a.shape as [number, number, number],
          output.shape as [number, number, number],
          env().get('WEBGPU_MATMUL_WORK_PER_THREAD') as number);
    } else {
      program = new MatMulPackedProgram(
          a.shape as [number, number, number],
          output.shape as [number, number, number],
          env().get('WEBGPU_MATMUL_WORK_PER_THREAD') as number, transposeA,
          transposeB);
    }

    return backend.compileAndRun(program, [a, b], output);
}
