/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {backend_util, TensorInfo} from '@tensorflow/tfjs-core';
import {activationFnSnippet} from './activation_util';
import {matMulReadWriteFnSource} from './matmul_packed_webgpu';
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';

export function makeMatMulOptimizedSource(
    workGroupSize: [number, number, number]): string {
  const tileAOuter = workGroupSize[1];
  const tileBOuter = workGroupSize[0];
  const tileInner = workGroupSize[0];
  return `
  var<workgroup> mm_Asub : array<array<vec4<f32>, ${tileInner}>, ${tileAOuter}>;
  var<workgroup> mm_Bsub : array<array<vec4<f32>, ${tileBOuter}>, ${tileInner}>;

  ${main()} {
    let tileRow = i32(localId.y);
    let tileCol = i32(localId.x);
    let globalRow = i32(globalId.y);
    let globalCol = i32(globalId.x);
    let batch = i32(globalId.z);

    // uniforms.dimInner should be greater than 0.
    let numTiles = (uniforms.dimInner - 1) / (${tileInner} * 4) + 1;
    var acc = vec4<f32>(0.0);

    var globalColA = tileCol;
    var globalRowB = tileRow;

    for (var t = 0; t < numTiles; t = t + 1) {
      mm_Asub[tileRow][tileCol] = mm_readA(batch, globalRow, globalColA);
      globalColA = globalColA + ${tileInner};
      for (var t1 = 0; t1 < 4; t1 = t1 + 1)
      {
        mm_Bsub[tileRow][tileCol] = mm_readB(batch, globalRowB, globalCol);
        globalRowB = globalRowB + ${tileInner};
        workgroupBarrier();
        var ACached = mm_Asub[tileRow][t1 * 2];
        var BCached0 = mm_Bsub[0][tileCol];
        var BCached1 = mm_Bsub[1][tileCol];
        var BCached2 = mm_Bsub[2][tileCol];
        var BCached3 = mm_Bsub[3][tileCol];
        var dot0 = dot(ACached, vec4<f32>(BCached0[0], BCached1[0], BCached2[0], BCached3[0]));
        var dot1 = dot(ACached, vec4<f32>(BCached0[1], BCached1[1], BCached2[1], BCached3[1]));
        var dot2 = dot(ACached, vec4<f32>(BCached0[2], BCached1[2], BCached2[2], BCached3[2]));
        var dot3 = dot(ACached, vec4<f32>(BCached0[3], BCached1[3], BCached2[3], BCached3[3]));
        acc = acc + vec4<f32>(dot0, dot1, dot2, dot3);

        ACached = mm_Asub[tileRow][t1 * 2 + 1];
        BCached0 = mm_Bsub[4][tileCol];
        BCached1 = mm_Bsub[5][tileCol];
        BCached2 = mm_Bsub[6][tileCol];
        BCached3 = mm_Bsub[7][tileCol];
        dot0 = dot(ACached, vec4<f32>(BCached0[0], BCached1[0], BCached2[0], BCached3[0]));
        dot1 = dot(ACached, vec4<f32>(BCached0[1], BCached1[1], BCached2[1], BCached3[1]));
        dot2 = dot(ACached, vec4<f32>(BCached0[2], BCached1[2], BCached2[2], BCached3[2]));
        dot3 = dot(ACached, vec4<f32>(BCached0[3], BCached1[3], BCached2[3], BCached3[3]));
        acc = acc + vec4<f32>(dot0, dot1, dot2, dot3);
        workgroupBarrier();
      }
    }

    mm_write(batch, globalRow, globalCol, acc);
  }
  `;
}

export class MatMulOptimizedProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  variableTypes: string[];
  uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workGroupSize: [number, number, number] = [8, 8, 1];
  transposeA: boolean;
  transposeB: boolean;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  batchAEqualOne: boolean;
  batchBEqualOne: boolean;
  isVec4?: boolean;

  constructor(
      aShape: [number, number, number], bShape: [number, number, number],
      outputShape: [number, number, number], transposeA = false,
      transposeB = false, bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;

    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    this.dispatch = [
      Math.ceil(outputShape[2] / (this.workGroupSize[0] * 4)),
      Math.ceil(outputShape[1] / (this.workGroupSize[1])), outputShape[0]
    ];
    this.isVec4 = true;

    const addBias = bias != null;
    if (addBias) {
      this.variableNames.push('bias');
    }

    const hasPreluActivationWeights = preluActivationWeights != null;
    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.transposeA = transposeA;
    this.transposeB = transposeB;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.batchAEqualOne = aShape[0] === 1;
    this.batchBEqualOne = bShape[0] === 1;
    this.shaderKey = `matMulOptimizedNew_${this.activation}_${transposeA}_${
        transposeB}_${this.batchAEqualOne}_${this.batchBEqualOne}`;
  }

  getUserCode(): string {
    const userCode = `
      ${
        activationFnSnippet(
            this.activation, this.hasPreluActivationWeights, true)}
      ${
        matMulReadWriteFnSource(
            this.addBias, this.activation, this.batchAEqualOne,
            this.batchBEqualOne, this.transposeA, this.transposeB, false, false,
            false, 4)}
      ${makeMatMulOptimizedSource(this.workGroupSize)}
    `;
    return userCode;
  }
}
