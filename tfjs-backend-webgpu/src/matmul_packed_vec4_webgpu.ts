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

import {getMainHeaderString} from './shader_preprocessor';
import {computeDispatch, tilesFitEvenlyIntoShape} from './webgpu_util';

import {mapActivationToShaderProgram} from './activation_util';
import {WebGPUProgram} from './webgpu_program';

export function makeMatMulPackedVec4Source(workPerThread: number[],
    tileAOuter: number, tileBOuter: number, tileInner: number): string {
  util.assert(
    tileInner % 4 === 0 && workPerThread[0] === 4,
      () => 'tileInner must be divisible by 4. And ColPerThread must be 4');
  return `
  var<workgroup> mm_Asub : array<array<vec4<f32>, ${
      tileInner / workPerThread[0]}>, ${tileAOuter}>;
  var<workgroup> mm_Bsub : array<array<vec4<f32>, ${
      tileBOuter / workPerThread[0]}>, ${tileInner}>;

  let RowPerThread = ${workPerThread[1]};
  let ColPerThread = ${workPerThread[0]};
  let TileInner = ${tileInner};

  ${getMainHeaderString()}

    let tileRow = ${tileAOuter === 1 ? '0' : 'i32(localId.y) * RowPerThread'};
    let tileCol = i32(localId.x);

    let globalRow = ${
        tileAOuter === 1 ? '0' : 'i32(globalId.y) * RowPerThread'};
    let globalCol = i32(globalId.x);
    let numTiles = (uniforms.dimInner - 1) / TileInner + 1;

    var acc: array<vec4<f32>, RowPerThread>;
    var ACached : vec4<f32>;
    var BCached : array<vec4<f32>, 4>;

    // Loop over shared dimension.
    var globalColA = tileCol;
    let RowPerThreadB = TileInner / i32(workGroupSizeY);
    let tileRowB = i32(localId.y) * RowPerThreadB;
    for (var t = 0; t < numTiles; t = t + 1) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < RowPerThread; innerRow = innerRow + 1) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileCol;
            mm_Asub[inputRow][inputCol] = mm_readA(globalRow + innerRow, globalColA, globalId);
        }
        globalColA = globalColA + TileInner / ColPerThread;

        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < RowPerThreadB; innerRow = innerRow + 1) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(t * TileInner + inputRow, globalCol, globalId);
        }

        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < TileInner / ColPerThread; k = k + 1) {
            BCached[0] = mm_Bsub[k * ColPerThread][tileCol];
            BCached[1] = mm_Bsub[k * ColPerThread + 1][tileCol];
            BCached[2] = mm_Bsub[k * ColPerThread + 2][tileCol];
            BCached[3] = mm_Bsub[k * ColPerThread + 3][tileCol];

            for (var i = 0; i < RowPerThread; i = i + 1) {
                ACached = mm_Asub[tileRow + i][k];
                acc[i] = BCached[0] * ACached.x + acc[i];
                acc[i] = BCached[1] * ACached.y + acc[i];
                acc[i] = BCached[2] * ACached.z + acc[i];
                acc[i] = BCached[3] * ACached.w + acc[i];
            }
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
  uniforms = `dimAOuter : i32; dimBOuter : i32; dimInner : i32;`;
  workGroupSize: [number, number, number] = [8, 8, 1];
  elementsPerThread: [number, number, number];
  isVec4 = true;
  aShape: [number, number, number];
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  tileAOuter:number;
  tileBOuter:number;
  tileInner:number;
  fitA: boolean;
  fitB: boolean;

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      rowPerThread: number, bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    // The first element in elementsPerThread must be 4.
    if (outputShape[1] === 1) {
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

    this.tileAOuter = outputShape[1] === 1 ? 1 :
        this.workGroupSize[1] * this.elementsPerThread[1];
    this.tileBOuter = this.workGroupSize[0] * this.elementsPerThread[0];
    this.tileInner = this.tileBOuter;

    this.aShape = aShape;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;

    [this.fitA, this.fitB] = this.getShapeFit();

    this.shaderKey = `matMulPackedVec4_${this.activation}_${
        this.fitA}_${this.fitB}_${this.elementsPerThread}`;
  }

  getShapeFit(): boolean[] {
    const dimInner = this.aShape[2];
    const dimBOuter = this.outputShape[2];
    const bShape = [this.outputShape[0], dimInner, dimBOuter];

    const tileSizeA = [this.tileAOuter, this.tileInner];
    const tileSizeB = [this.tileInner, this.tileBOuter];
    return [
      tilesFitEvenlyIntoShape(tileSizeA, this.aShape.slice(1)),
      tilesFitEvenlyIntoShape(tileSizeB, bShape.slice(1))
    ];
  }

  getUserCode(): string {
    const sampleA = this.fitA ?
        `return A.numbers[batch * batchASize + row * uniforms.dimInner / 4 + col]` :
        `if (coordsInBounds2D(vec2<i32>(row, col * 4), vec2<i32>(uniforms.dimAOuter, uniforms.dimInner))) {
            return A.numbers[batch * batchASize + row * uniforms.dimInner / 4 + col];
        }
        return vec4<f32>(0.0)`;

    const sampleB = this.fitB ?
        `return B.numbers[batch * batchBSize + row * uniforms.dimBOuter / 4 + col]` :
        `if(coordsInBounds2D(vec2<i32>(row, col * 4), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
             return B.numbers[batch * batchBSize + row * uniforms.dimBOuter / 4 + col];
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
    const addBiasSnippet = this.addBias ?
        'value = value + getBiasByOutputCoords(outCoord);' :
        '';

    const userCode = `
      ${activationSnippet}
      fn mm_readA(row : i32, col : i32,  globalId : vec3<u32>) -> vec4<f32> {
        let batchASize = uniforms.aShape[1] * uniforms.aShape[2] / 4;
        let batch = i32(globalId.z);
        ${sampleA};
      }

      fn mm_readB(row : i32, col : i32,  globalId : vec3<u32>) -> vec4<f32> {
        let batchBSize = uniforms.bShape[1] * uniforms.bShape[2] / 4;
        let batch = i32(globalId.z);
        ${sampleB};
      }

      fn mm_write(row : i32, col : i32, valueIn : vec4<f32>, globalId : vec3<u32>) {
        if (row < uniforms.aShape[1] && col * 4 < uniforms.bShape[2])
        {
          var value = valueIn;
          let batch = i32(globalId.z);
          let outCoord = vec3<i32>(batch, row, col * 4);
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutputAtCoords(outCoord[0], outCoord[1], outCoord[2], value);
        }
      }
      ${makeMatMulPackedVec4Source(this.elementsPerThread,
          this.tileAOuter, this.tileBOuter, this.tileInner)}
    `;

    return userCode;
  }
}
