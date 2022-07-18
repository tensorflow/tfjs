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

import {backend_util, TensorInfo, util} from '@tensorflow/tfjs-core';
import {activationFnSnippet, biasActivationSnippet} from './activation_util';
import {getMainHeaderString, WebGPUProgram} from './webgpu_program';

export function makeMatMulSmallOutputSizeSource(
    workGroupSize: [number, number, number]): string {
  const tileAOuter = workGroupSize[1];
  const tileBOuter = workGroupSize[0];
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
  ${getMainHeaderString()}
    let tileRow = i32(localId.y);
    let tileCol = i32(localId.x);
    let globalRow = i32(globalId.y);
    let globalCol = i32(globalId.x);

    // uniforms.dimInner should be greater than 0.
    let numTiles = (uniforms.dimInner - 1) / ${tileInner} + 1;
    var acc = 0.0;

    var globalColA = tileCol;
    var globalRowB = 0;
    var regA = mm_readA(globalRow, globalColA, globalId);
    var regB0 = mm_readB(globalRowB + 2 * tileRow, globalCol, globalId);
    var regB1 = mm_readB(globalRowB + 2 * tileRow + 1, globalCol, globalId);
    globalColA = globalColA + ${tileInner};
    globalRowB = globalRowB + ${tileInner};

    for (var t = 0; t < numTiles; t = t + 1) {
      mm_Asub[tileRow][tileCol] = regA;
      mm_Bsub[2 * tileRow][tileCol] = regB0;
      mm_Bsub[2 * tileRow + 1][tileCol] = regB1;

      workgroupBarrier();

      regA = mm_readA(globalRow, globalColA, globalId);
      regB0 = mm_readB(globalRowB + 2 * tileRow, globalCol, globalId);
      regB1 = mm_readB(globalRowB + 2 * tileRow + 1, globalCol, globalId);
      globalColA = globalColA + ${tileInner};
      globalRowB = globalRowB + ${tileInner};

      for (var k = 0; k < ${tileInner}; k = k + 1) {
        acc = acc + mm_Asub[tileRow][k] * mm_Bsub[k][tileCol];
      }
      workgroupBarrier();
    }

    mm_write(globalRow, globalCol, acc, globalId);
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
  workGroupSize: [number, number, number] = [16, 8, 1];
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  batchAEqualOne: boolean;
  batchBEqualOne: boolean;

  constructor(
      aShape: [number, number, number], bShape: [number, number, number],
      outputShape: [number, number, number], bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    util.assert(
        aShape[1] <= 16 || bShape[2] <= 16,
        () =>
            'This program can be only used when A width or B Height are small');

    this.outputShape = outputShape;

    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    this.dispatch = [
      Math.ceil(outputShape[2] / this.workGroupSize[0]),
      Math.ceil(outputShape[1] / this.workGroupSize[1]), outputShape[0]
    ];

    const addBias = bias != null;
    if (addBias) {
      this.variableNames.push('bias');
    }

    const hasPreluActivationWeights = preluActivationWeights != null;
    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.batchAEqualOne = aShape[0] === 1;
    this.batchBEqualOne = bShape[0] === 1;
    this.shaderKey = `matMulSmallOutputSize_${this.activation}_${
        this.batchAEqualOne}_${this.batchBEqualOne}`;
  }

  getUserCode(): string {
    const sampleA = `var result: f32;
        if (coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimAOuter, uniforms.dimInner))) {
          result =  A[batch * batchASize + row * uniforms.dimInner + col];
        } else {
          result = 0.0;
        }
        return result;`;

    const sampleB = `var result: f32;
        if (coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
           result = B[batch * batchBSize + row * uniforms.dimBOuter + col];
         } else {
           result = 0.0;
         }
         return result;`;

    const userCode = `
      ${activationFnSnippet(this.activation, this.hasPreluActivationWeights)}

      fn mm_readA(row : i32, col : i32,  globalId : vec3<u32>) -> f32 {
        ${
        this.batchAEqualOne ? `
          let batch = 0;
          let batchASize = 0;
          ` :
                              `
          let batchASize = uniforms.aShape[1] * uniforms.aShape[2];
          let batch = i32(globalId.z);
          `}
        ${sampleA}
      }
      fn mm_readB(row : i32, col : i32,  globalId : vec3<u32>) -> f32 {
        ${
        this.batchBEqualOne ? `
          let batch = 0;
          let batchBSize = 0;
          ` :
                              `
          let batch = i32(globalId.z);
          let batchBSize = uniforms.bShape[1] * uniforms.bShape[2];
          `}
        ${sampleB}
      }
      fn mm_write(row : i32, col : i32, valueIn : f32, globalId : vec3<u32>) {
        if (coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimAOuter, uniforms.dimBOuter))) {
          let batch = i32(globalId.z);
          let coords = vec3<i32>(batch, row, col);
          var value = valueIn;
          ${biasActivationSnippet(this.addBias, this.activation)}
          setOutputAtCoords(batch, row, col, value);
        }
      }
      ${makeMatMulSmallOutputSizeSource(this.workGroupSize)}
    `;
    return userCode;
  }
}
