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

import {backend_util, TensorInfo} from '@tensorflow/tfjs-core';
import {activationFnSnippet} from './activation_util';
import {matMulReadWriteFnSource} from './matmul_packed_webgpu';
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';

export function makeMatMulSmallOutputSizeSource(
    workgroupSize: [number, number, number]): string {
  const tileAOuter = workgroupSize[1];
  const tileBOuter = workgroupSize[0];
  const tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
  return `
  var<workgroup> mm_Asub : array<array<f32, ${tileInner}>, ${tileAOuter}>;
  var<workgroup> mm_Bsub : array<array<f32, ${tileBOuter}>, ${tileInner}>;

  // If the output size is small for matrix multiplication, avoid to use vec4
  // and handle some elements per thread to optimally utilize the ALU.
  // Read data from global memory to registers firstly, then store them into
  // shared memory, so it is instruction-Level parallelism for arithmetic
  // operations and others handle IO operations between barrier api, makes ALU
  // and load/store units work simultaneously, could improves the performance.
  ${main()} {
    let tileRow = i32(localId.y);
    let tileCol = i32(localId.x);
    let globalRow = i32(globalId.y);
    let globalCol = i32(globalId.x);
    let batch = i32(globalId.z);
    let batchA = batch % uniforms.aShape[0];
    let batchB = batch % uniforms.bShape[0];

    // uniforms.dimInner should be greater than 0.
    let numTiles = (uniforms.dimInner - 1) / ${tileInner} + 1;
    var acc = 0.0;

    var globalColA = tileCol;
    var globalRowB = 0;
    var regA = mm_readA(batchA, globalRow, globalColA);
    var regB0 = mm_readB(batchB, globalRowB + 2 * tileRow, globalCol);
    var regB1 = mm_readB(batchB, globalRowB + 2 * tileRow + 1, globalCol);
    globalColA = globalColA + ${tileInner};
    globalRowB = globalRowB + ${tileInner};

    for (var t = 0; t < numTiles; t = t + 1) {
      mm_Asub[tileRow][tileCol] = regA;
      mm_Bsub[2 * tileRow][tileCol] = regB0;
      mm_Bsub[2 * tileRow + 1][tileCol] = regB1;

      workgroupBarrier();

      regA = mm_readA(batchA, globalRow, globalColA);
      regB0 = mm_readB(batchB, globalRowB + 2 * tileRow, globalCol);
      regB1 = mm_readB(batchB, globalRowB + 2 * tileRow + 1, globalCol);
      globalColA = globalColA + ${tileInner};
      globalRowB = globalRowB + ${tileInner};

      for (var k = 0; k < ${tileInner}; k = k + 1) {
        acc = acc + mm_Asub[tileRow][k] * mm_Bsub[k][tileCol];
      }
      workgroupBarrier();
    }

    mm_write(batch, globalRow, globalCol, acc);
  }
  `;
}

export class MatMulSmallOutputSizeProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workgroupSize: [number, number, number] = [16, 8, 1];
  transposeA: boolean;
  transposeB: boolean;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;

  constructor(
      aShape: [number, number, number], bShape: [number, number, number],
      outputShape: [number, number, number], transposeA = false,
      transposeB = false, bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;

    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    this.dispatch = [
      Math.ceil(outputShape[2] / this.workgroupSize[0]),
      Math.ceil(outputShape[1] / this.workgroupSize[1]), outputShape[0]
    ];

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
    this.shaderKey =
        `matMulSmallOutputSize_${this.activation}_${transposeA}_${transposeB}`;
  }

  getUserCode(): string {
    const userCode = `
      ${activationFnSnippet(this.activation, this.hasPreluActivationWeights)}
      ${
        matMulReadWriteFnSource(
            this.addBias, this.activation, this.transposeA, this.transposeB)}
      ${makeMatMulSmallOutputSizeSource(this.workgroupSize)}
    `;
    return userCode;
  }
}
