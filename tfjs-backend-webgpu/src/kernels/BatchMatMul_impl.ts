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

import {backend_util, env, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';

import {MatMulPackedVec4Program} from './matmul_packed_vec4_webgpu';
import {MatMulPackedProgram} from './matmul_packed_webgpu';
import {reshape} from './Reshape';

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
  const aRank = a.shape.length;
  const bRank = b.shape.length;

  const innerShapeA = transposeA ? a.shape[aRank - 2] : a.shape[aRank - 1];
  const innerShapeB = transposeB ? b.shape[bRank - 1] : b.shape[bRank - 2];

  const outerShapeA = transposeA ? a.shape[aRank - 1] : a.shape[aRank - 2];
  const outerShapeB = transposeB ? b.shape[bRank - 2] : b.shape[bRank - 1];

  const outerDimsA = a.shape.slice(0, -2);
  const outerDimsB = b.shape.slice(0, -2);

  const batchDimA = util.sizeFromShape(outerDimsA);
  const batchDimB = util.sizeFromShape(outerDimsB);

  const batchDimsCompatible =
      batchDimA === batchDimB || batchDimA === 1 || batchDimB === 1;

  util.assert(
      aRank >= 2 && bRank >= 2 && batchDimsCompatible,
      () => `Error in matMul: the input batch dimensions must either be the ` +
          `same or at least one input batch dimension must be 1. Got input ` +
          `batch dimensions of (${outerDimsA}) and (${outerDimsB}).`);

  const outShapeOuterDims =
      batchDimA > batchDimB ? a.shape.slice(0, -2) : b.shape.slice(0, -2);
  const outShape = outShapeOuterDims.concat([outerShapeA, outerShapeB]);

  util.assert(
      innerShapeA === innerShapeB,
      () => `Error in matMul: inner shapes (${innerShapeA}) and (` +
          `${innerShapeB}) of Tensors with shapes ${a.shape} and ` +
          `${b.shape} and transposeA=${transposeA}` +
          ` and transposeB=${transposeB} must match.`);

  const a3dShape: [number, number, number] = transposeA ?
      [batchDimA, innerShapeA, outerShapeA] :
      [batchDimA, outerShapeA, innerShapeA];
  const b3dShape: [number, number, number] = transposeB ?
      [batchDimB, outerShapeB, innerShapeB] :
      [batchDimB, innerShapeB, outerShapeB];

  // The rest of the implementation is designed to operate on rank-3 tensors
  const a3d = reshape({inputs: {x: a}, backend, attrs: {shape: a3dShape}});
  const b3d = reshape({inputs: {x: b}, backend, attrs: {shape: b3dShape}});
  const intermediates: TensorInfo[] = [a3d, b3d];

  const batchDim = Math.max(batchDimA, batchDimB);

  const hasBias = bias != null;
  const hasPreluActivationWeights = preluActivationWeights != null;
  const useVec4 = a.shape[2] % 4 === 0 && b.shape[2] % 4 === 0 && !transposeA &&
      !transposeB;
  const fusedActivation = activation ?
      backend.mapActivationToShaderProgram(activation, useVec4) :
      null;
  let program: MatMulPackedProgram|MatMulPackedVec4Program;
  if (useVec4) {
    // TODO: Currently we need to make sure that a.shape[2] and b.shape[2]
    // are divisible by 4 since we use vec4 to get data. In future, we can
    // remove this limitation by insert 0 to pack data.
    program = new MatMulPackedVec4Program(
        a3dShape, [batchDim, outerShapeA, outerShapeB],
        env().get('WEBGPU_MATMUL_WORK_PER_THREAD') as number, hasBias,
        fusedActivation, hasPreluActivationWeights);
  } else {
    program = new MatMulPackedProgram(
        a3dShape, [batchDim, outerShapeA, outerShapeB],
        env().get('WEBGPU_MATMUL_WORK_PER_THREAD') as number, transposeA,
        transposeB, hasBias, fusedActivation, hasPreluActivationWeights);
  }
  const inputs: TensorInfo[] = [a3d, b3d];
  if (bias) {
    inputs.push(bias);
  }
  if (preluActivationWeights) {
    inputs.push(preluActivationWeights);
  }
  const out = backend.runWebGPUProgram(program, inputs, a.dtype);
  const outReshaped =
      reshape({inputs: {x: out}, backend, attrs: {shape: outShape}});
  intermediates.push(out);
  for (const i of intermediates) {
    backend.disposeData(i.dataId);
  }
  return outReshaped;
}
