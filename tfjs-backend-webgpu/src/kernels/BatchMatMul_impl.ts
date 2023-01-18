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

import {backend_util, broadcast_util, env, TensorInfo, util} from '@tensorflow/tfjs-core';

import {WebGPUBackend} from '../backend_webgpu';
import {MatMulPackedProgram} from '../matmul_packed_webgpu';
import {MatMulReduceProgram} from '../matmul_reduce_webgpu';
import {MatMulSmallOutputSizeProgram} from '../matmul_small_output_size_webgpu';
import {BiasActivationProgram, MatMulSplitKProgram} from '../matmul_splitK_webgpu';
import {WebGPUProgram} from '../webgpu_program';
import {MatMulProgramType} from '../webgpu_util';

import {fill} from './Fill';
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

  const outShapeOuterDims = broadcast_util.assertAndGetBroadcastShape(
      a.shape.slice(0, -2), b.shape.slice(0, -2));
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

  const inputs: TensorInfo[] = [a3d, b3d];
  const dimensions = [
    {type: 'int32', data: [outerShapeA]}, {type: 'int32', data: [outerShapeB]},
    {type: 'int32', data: [innerShapeA]}
  ];

  let program: WebGPUProgram;
  let out: TensorInfo;
  const outputShape: [number, number, number] =
      [batchDim, outerShapeA, outerShapeB];
  let matmulProgramType = env().get('WEBGPU_MATMUL_PROGRAM_TYPE') as number;
  if (matmulProgramType < 0) {
    // Usually increasing workgroups is a good way to gain more performance for
    // few workgroups by tiling 32x32 (default matmul algorithm). Currently,
    // there are three ways to increase workgroups. 1) MatMulReduceProgram,
    // which is used only when the output size is very small (128 for now). 2)
    // MatMulSplitKProgram, increasing workgroups by spliting K. 3)
    // MatMulSmallOutputSizeProgram, increasing workgroups by small tile size.
    // For different devices, the minimum optimal workgroups may be different.
    // So here we set a |thresholdToIncreaseWorkgroups| to indicate whether we
    // need to increase workgroups. And the literal number is an empirical
    // value.
    const thresholdFlagValue =
        env().getNumber('WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL');
    const thresholdToIncreaseWorkgroups = thresholdFlagValue > 0 ?
        thresholdFlagValue :
        backend.thresholdToIncreaseWorkgroups;
    const workgroupsBy32x32 =
        batchDim * Math.ceil(outerShapeA / 32) * Math.ceil(outerShapeB / 32);
    const hasFewWorkgroups =
        workgroupsBy32x32 <= thresholdToIncreaseWorkgroups ||
        (outerShapeA <= 8 &&
         workgroupsBy32x32 <= thresholdToIncreaseWorkgroups * 2);
    if (hasFewWorkgroups) {
      if (batchDim * outerShapeA * outerShapeB <= 128) {
        matmulProgramType = MatMulProgramType.MatMulReduceProgram;
      } else if (batchDim === 1 && innerShapeB >= 2000) {
        matmulProgramType = MatMulProgramType.MatMulSplitKProgram;
      } else {
        matmulProgramType = MatMulProgramType.MatMulSmallOutputSizeProgram;
      }
    } else {
      matmulProgramType = MatMulProgramType.MatMulPackedProgram;
    }
  }

  switch (matmulProgramType) {
    case MatMulProgramType.MatMulReduceProgram:
      program = new MatMulReduceProgram(
          outputShape, transposeA, transposeB, bias, activation,
          preluActivationWeights);
      break;
    case MatMulProgramType.MatMulSplitKProgram: {
      // The output buffer must be initailzed to zero before using since we
      // use atomicAdd in MatMulSplitKProgram.
      out = fill(
          {backend, attrs: {shape: outputShape, value: 0, dtype: a.dtype}});
      program = new MatMulSplitKProgram(
          outputShape, innerShapeB, transposeA, transposeB);
      if (bias || activation) {
        out =
            backend.runWebGPUProgram(program, inputs, a.dtype, dimensions, out);
        const biasActivationProgram = new BiasActivationProgram(
            out.shape, bias, activation, preluActivationWeights);
        let uniformData = null;
        const activationInputs: TensorInfo[] = [out];
        if (bias) {
          activationInputs.push(bias);
        }
        if (preluActivationWeights) {
          activationInputs.push(preluActivationWeights);
        }
        if (activation === 'leakyrelu') {
          uniformData = [{type: 'float32', data: [leakyreluAlpha]}];
          biasActivationProgram.uniforms += ' alpha : f32,';
        }
        const outActivated = backend.runWebGPUProgram(
            biasActivationProgram, activationInputs, out.dtype, uniformData);
        intermediates.push(out);
        const outReshaped = reshape(
            {inputs: {x: outActivated}, backend, attrs: {shape: outShape}});
        intermediates.push(outActivated);
        for (const i of intermediates) {
          backend.disposeData(i.dataId);
        }
        return outReshaped;
      }
      break;
    }
    case MatMulProgramType.MatMulSmallOutputSizeProgram:
      program = new MatMulSmallOutputSizeProgram(
          a3dShape, b3dShape, outputShape, transposeA, transposeB, bias,
          activation, preluActivationWeights);
      break;
    case MatMulProgramType.MatMulPackedProgram:
      // Experiments show that sequential access is more friendly for Intel
      // GPUs.
      const sequentialAccessByThreads = backend.adapterInfo.isIntel();
      program = new MatMulPackedProgram(
          a3dShape, outputShape, transposeA, transposeB, bias, activation,
          preluActivationWeights, sequentialAccessByThreads);
      break;
    default:
      throw new Error(`Unsupported MatMulProgramType ${matmulProgramType}.`);
  }

  if (bias) {
    inputs.push(bias);
  }
  if (preluActivationWeights) {
    inputs.push(preluActivationWeights);
  }
  if (activation === 'leakyrelu') {
    dimensions.push({type: 'float32', data: [leakyreluAlpha]});
    program.uniforms += ' alpha : f32,';
  }
  out = backend.runWebGPUProgram(program, inputs, a.dtype, dimensions, out);
  const outReshaped =
      reshape({inputs: {x: out}, backend, attrs: {shape: outShape}});
  intermediates.push(out);
  for (const i of intermediates) {
    backend.disposeData(i.dataId);
  }
  return outReshaped;
}
