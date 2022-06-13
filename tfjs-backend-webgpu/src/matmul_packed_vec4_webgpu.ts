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

import {backend_util, TensorInfo, util} from '@tensorflow/tfjs-core';
import {mapActivationToShaderProgram} from './activation_util';
import {WebGPUProgram} from './webgpu_program';
import {computeDispatch} from './webgpu_util';

const writeDataToSubASnippet =
    (transpose: boolean, innerAElementSize: number) => {
      if (transpose) {
        return `
        mm_Asub[inputRow][inputCol] = mm_readA(
          t * TileInner + inputRow,
          globalRowStart / ${innerAElementSize} + inputCol, globalId);
        `;

      } else {
        return `
        mm_Asub[inputRow][inputCol] = mm_readA(
          globalRow + innerRow,
          t * TileInner / ${innerAElementSize} + inputCol, globalId);
        `;
      }
    };

const calculateResultSnippet =
    (transposeA: boolean, innerElementSize: number) => {
      if (transposeA) {
        return `
        let ACached0 = mm_Asub[k * InnerElementSize][localRow];
        let ACached1 = mm_Asub[k * InnerElementSize + 1][localRow];
        let ACached2 = mm_Asub[k * InnerElementSize + 2][localRow];
        ${
            innerElementSize === 3 ?
                '' :
                'let ACached3 = mm_Asub[k * InnerElementSize + 3][localRow];'}
        for (var i = 0; i < RowPerThread; i = i + 1) {
          acc[i] = BCached[0] * ACached0[i] + acc[i];
          acc[i] = BCached[1] * ACached1[i] + acc[i];
          acc[i] = BCached[2] * ACached2[i] + acc[i];
          ${
            innerElementSize === 3 ?
                '' :
                'acc[i] = BCached[3] * ACached3[i] + acc[i];'}
        }`;
      } else {
        return `
        for (var i = 0; i < RowPerThread; i = i + 1) {
          let ACached = mm_Asub[tileRow + i][k];
          acc[i] = BCached[0] * ACached.x + acc[i];
          acc[i] = BCached[1] * ACached.y + acc[i];
          acc[i] = BCached[2] * ACached.z + acc[i];
          ${
            innerElementSize === 3 ?
                '' :
                'acc[i] = BCached[3] * ACached.w + acc[i];'}
        }`;
      }
    };

export function makeMatMulPackedVec4Source(
    workPerThread: number[], tileAOuter: number, tileBOuter: number,
    tileInner: number, innerElementSize = 4, transposeA = false): string {
  const tileAWidth = transposeA ? tileAOuter : tileInner;
  const tileAHight = transposeA ? tileInner : tileAOuter;
  const innerAElementSize = transposeA ? workPerThread[1] : innerElementSize;
  // For simplicity, if transposeA is true, tileAOuter must be equal to
  // tileBOuter.
  util.assert(
      ((transposeA && tileAOuter === tileBOuter) ||
       (tileInner % 4 === 0 || tileInner % 3 === 0)) &&
          workPerThread[0] === 4 &&
          (innerElementSize === 3 || innerElementSize === 4),
      () => `tileInner ${tileInner} must be divisible by 4|3. ColPerThread ${
          workPerThread[0]} must be 4.
           innerElementSize ${innerElementSize} must be 3|4.`);
  return `
  var<workgroup> mm_Asub : array<array<vec${innerAElementSize}<f32>, ${
      tileAWidth / innerAElementSize}>, ${tileAHight}>;
  var<workgroup> mm_Bsub : array<array<vec4<f32>, ${
      tileBOuter / workPerThread[0]}>, ${tileInner}>;

  let RowPerThread = ${workPerThread[1]};
  let ColPerThread = ${workPerThread[0]};
  let InnerElementSize = ${innerElementSize};
  let TileInner = ${tileInner};

  @stage(compute) @workgroup_size(workGroupSizeX, workGroupSizeY, workGroupSizeZ)
  fn main(@builtin(local_invocation_id) LocalId : vec3<u32>,
          @builtin(global_invocation_id) GlobalId : vec3<u32>,
          @builtin(num_workgroups) NumWorkgroups: vec3<u32>,
          @builtin(workgroup_id) workgroupId: vec3<u32>) {
    localId = LocalId;
    globalId = GlobalId;
    numWorkgroups = NumWorkgroups;

    let localRow = i32(localId.y);
    let tileRow = ${tileAOuter === 1 ? '0' : 'localRow * RowPerThread'};
    let tileCol = i32(localId.x);

    let globalRow = ${
      tileAOuter === 1 ? '0' : 'i32(globalId.y) * RowPerThread'};
    let globalCol = i32(globalId.x);
    let globalRowStart = i32(workgroupId.y) * ${tileAOuter};

    let numTiles = (uniforms.dimInner - 1) / TileInner + 1;

    var acc: array<vec4<f32>, RowPerThread>;
    var BCached : array<vec4<f32>, 4>;

    // Loop over shared dimension.
    let RowPerThreadB = TileInner / i32(workGroupSizeY);
    let tileRowB = localRow * RowPerThreadB;
    for (var t = 0; t < numTiles; t = t + 1) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < RowPerThread; innerRow = innerRow + 1) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileCol;
            ${writeDataToSubASnippet(transposeA, innerAElementSize)}
        }

        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < RowPerThreadB; innerRow = innerRow + 1) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(t * TileInner + inputRow, globalCol, globalId);
        }

        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < TileInner / InnerElementSize; k = k + 1) {
            BCached[0] = mm_Bsub[k * InnerElementSize][tileCol];
            BCached[1] = mm_Bsub[k * InnerElementSize + 1][tileCol];
            BCached[2] = mm_Bsub[k * InnerElementSize + 2][tileCol];
            ${
      innerElementSize === 3 ?
          '' :
          'BCached[3] = mm_Bsub[k * InnerElementSize + 3][tileCol];'}

            ${calculateResultSnippet(transposeA, innerElementSize)}
        }

        workgroupBarrier();
    }

    for (var innerRow = 0; innerRow < RowPerThread; innerRow = innerRow + 1) {
        mm_write(globalRow + innerRow,
                 globalCol,
                 acc[innerRow], globalId);
    }
  }`;
}

export class MatMulPackedVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workGroupSize: [number, number, number] = [8, 8, 1];
  elementsPerThread: [number, number, number];
  isVec4 = true;
  aShape: [number, number, number];
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  tileAOuter: number;
  tileBOuter: number;
  tileInner: number;
  fitAOuter: boolean;
  fitBOuter: boolean;
  fitInner: boolean;
  batchAEqualOne: boolean;
  batchBEqualOne: boolean;
  transposeA: boolean;

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      batchAEqualOne: boolean, batchBEqualOne: boolean, transposeA = false,
      bias: TensorInfo = null, activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    // The first element in elementsPerThread must be 4.
    if (outputShape[1] === 1 && !transposeA) {
      this.elementsPerThread = [4, 1, 1];
    } else {
      this.elementsPerThread = [4, 4, 1];
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        this.elementsPerThread);

    const addBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.tileAOuter = outputShape[1] === 1 && !transposeA ?
        1 :
        this.workGroupSize[1] * this.elementsPerThread[1];
    this.tileBOuter = this.workGroupSize[0] * this.elementsPerThread[0];
    this.tileInner = this.tileBOuter;

    this.aShape = aShape;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.batchAEqualOne = batchAEqualOne;
    this.batchBEqualOne = batchBEqualOne;
    this.transposeA = transposeA;

    const dimInner = transposeA ? aShape[1] : aShape[2];
    this.fitAOuter = outputShape[1] % this.tileAOuter === 0;
    this.fitBOuter = outputShape[2] % this.tileBOuter === 0;
    this.fitInner = dimInner % this.tileInner === 0;

    this.shaderKey = `matMulPackedVec4_${this.activation}_${this.fitAOuter}_${
        this.fitBOuter}_${this.fitInner}_${this.elementsPerThread}_${
        this.batchAEqualOne}_${this.batchBEqualOne}_${this.transposeA}`;
  }

  getUserCode(): string {
    const sampleA = this.fitAOuter && this.fitInner ?
        `return A[batch * batchASize + row * uniforms.aShape[2] / 4 + col]` :
        `if (coordsInBounds2D(vec2<i32>(row, col * 4), vec2<i32>(uniforms.aShape[1], uniforms.aShape[2]))) {
            return A[batch * batchASize + row * uniforms.aShape[2] / 4 + col];
        }
        return vec4<f32>(0.0)`;

    const sampleB = this.fitInner && this.fitBOuter ?
        `return B[batch * batchBSize + row * uniforms.dimBOuter / 4 + col]` :
        `if(coordsInBounds2D(vec2<i32>(row, col * 4), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
             return B[batch * batchBSize + row * uniforms.dimBOuter / 4 + col];
        }
        return vec4<f32>(0.0)`;

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp =
          mapActivationToShaderProgram(this.activation, this.isVec4);
      if (this.hasPreluActivationWeights) {
        activationSnippet =
            `fn activation(a : vec4<f32>, outCoord : vec3<i32>) -> vec4<f32> {
                  let b = getPreluActivationWeightsByOutputCoords(outCoord);
                  ${activationOp}
                }`;
      } else {
        activationSnippet = `
            fn activation(a : vec4<f32>, outCoord : vec3<i32>) -> vec4<f32> {
              ${activationOp}
            }`;
      }

      applyActivationSnippet = 'value = activation(value, outCoord);';
    }
    const addBiasSnippet =
        this.addBias ? 'value = value + getBiasByOutputCoords(outCoord);' : '';

    const userCode = `
      ${activationSnippet}
      fn mm_readA(row : i32, col : i32,  globalId : vec3<u32>) -> vec4<f32> {
        ${
        this.batchAEqualOne ? `
          let batchASize = 0;
          let batch = 0;
        ` :
                              `
          let batchASize = uniforms.aShape[1] * uniforms.aShape[2] / 4;
          let batch = i32(globalId.z);
        `}

        ${sampleA};
      }

      fn mm_readB(row : i32, col : i32,  globalId : vec3<u32>) -> vec4<f32> {
        ${
        this.batchBEqualOne ? `
          let batchBSize = 0;
          let batch = 0;
          ` :
                              `
          let batchBSize = uniforms.bShape[1] * uniforms.bShape[2] / 4;
          let batch = i32(globalId.z);
       `}
        ${sampleB};
      }

      fn mm_write(row : i32, col : i32, valueIn : vec4<f32>, globalId : vec3<u32>) {
        if (row < uniforms.dimAOuter && col * 4 < uniforms.dimBOuter)
        {
          var value = valueIn;
          let batch = i32(globalId.z);
          let outCoord = vec3<i32>(batch, row, col * 4);
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutputAtCoords(outCoord[0], outCoord[1], outCoord[2], value);
        }
      }
      ${
        makeMatMulPackedVec4Source(
            this.elementsPerThread, this.tileAOuter, this.tileBOuter,
            this.tileInner, 4, this.transposeA)}
    `;

    return userCode;
  }
}
