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

import {computeDispatch, computeWorkGroupSizeForMatMul, tilesFitEvenlyIntoShape} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export function makeMatMulPackedVec4Source(workPerThread: number[]): string {
  return `
    vec4 mm_readA(int row, int col);
    vec4 mm_readB(int row, int col);
    void mm_write(int row, int col, vec4 value);

    const int RowPerThread = ${workPerThread[1]};
    const int ColPerThread = ${
      workPerThread[0]}; // only support ColPerThread = 4
    const int TileAOuter = int(gl_WorkGroupSize.y) * RowPerThread;
    const int TileBOuter = int(gl_WorkGroupSize.x) * ColPerThread;
    const int TileInner = TileBOuter;

    shared vec4 mm_Asub[TileAOuter][TileInner / ColPerThread];
    shared vec4 mm_Bsub[TileInner][TileBOuter / ColPerThread];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
      int tileCol = int(gl_LocalInvocationID.x);

      int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
      int globalCol = int(gl_GlobalInvocationID.x);

      int numTiles = (dimInner - 1) / TileInner + 1;

      vec4 acc[RowPerThread];
      vec4 ACached;
      vec4 BCached[4];

      // Without this initialization strange values show up in acc.
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          acc[innerRow] = vec4(0.0, 0.0, 0.0, 0.0);
      }

      // Loop over shared dimension.
      int globalColA = tileCol;
      const int RowPerThreadB = TileInner / int(gl_WorkGroupSize.y);
      int tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
            int inputRow = tileRow + innerRow;
            int inputCol = tileCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                globalColA);
        }
        globalColA += TileInner / ColPerThread;

        // Load one tile of B into local memory.
        for (int innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
            int inputRow = tileRowB + innerRow;
            int inputCol = tileCol;

            mm_Bsub[inputRow][inputCol] = mm_readB(
              t * TileInner + inputRow,
              globalCol);
        }

        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < TileInner / ColPerThread; k++) {
          BCached[0] = mm_Bsub[k * ColPerThread][tileCol];
          BCached[1] = mm_Bsub[k * ColPerThread + 1][tileCol];
          BCached[2] = mm_Bsub[k * ColPerThread + 2][tileCol];
          BCached[3] = mm_Bsub[k * ColPerThread + 3][tileCol];

          for (int i = 0; i < RowPerThread; i++) {
            ACached = mm_Asub[tileRow + i][k];
            acc[i] = BCached[0] * ACached.x + acc[i];
            acc[i] = BCached[1] * ACached.y + acc[i];
            acc[i] = BCached[2] * ACached.z + acc[i];
            acc[i] = BCached[3] * ACached.w + acc[i];
          }
        }
        barrier();
      }

      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        mm_write(globalRow + innerRow,
          globalCol,
          acc[innerRow]);
      }
    }
  `;
}

export class MatMulPackedVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number] = [16, 16, 1];
  isVec4 = true;
  aShape: [number, number, number];
  addBias: boolean;
  activation: string;
  hasPreluActivationWeights: boolean;

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      rowPerThread: number, addBias = false, activation: string = null,
      hasPreluActivationWeights = false) {
    this.outputShape = outputShape;
    this.workGroupSize = computeWorkGroupSizeForMatMul(
        outputShape[1], aShape[2], outputShape[2]);
    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    const vecSize = 4;
    if (outputShape[1] === 1) {
      rowPerThread = 1;
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [vecSize, rowPerThread, 1]);

    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.workPerThread = rowPerThread;
    this.aShape = aShape;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.shaderKey = `matMulPackedVec4_${rowPerThread}_${activation}`;
  }

  getUserCode(): string {
    const dimInner = this.aShape[2];
    const dimBOuter = this.outputShape[2];
    const bShape = [this.outputShape[0], dimInner, dimBOuter];
    const vecSize = 4;
    const tileAOuter = this.workGroupSize[1] * this.workPerThread;
    const tileBOuter = this.workGroupSize[0] * vecSize;
    const tileInner = tileBOuter;  // Make sure tileInner is divisible by 4.

    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];
    const fitA = tilesFitEvenlyIntoShape(tileSizeA, this.aShape.slice(1));
    const batchASize = this.aShape[1] * this.aShape[2] / vecSize;
    const batchBSize = bShape[1] * bShape[2] / vecSize;

    const sampleA = fitA ?
        `A[batch * ${batchASize} + row * dimInner / 4 + col]` :
        `coordsInBounds(ivec2(row, col * 4), ivec2(dimAOuter, dimInner)) ?
            A[batch * ${
            batchASize} + row * dimInner / 4 + col] : vec4(0.0, 0.0, 0.0, 0.0)`;

    const fitB = tilesFitEvenlyIntoShape(tileSizeB, bShape.slice(1));
    const sampleB = fitB ?
        `B[batch * ${batchBSize} + row * dimBOuter / 4 + col]` :
        `coordsInBounds(ivec2(row, col * 4), ivec2(dimInner, dimBOuter)) ?
            B[batch * ${batchBSize} + row * dimBOuter / 4 + col] :
            vec4(0.0, 0.0, 0.0, 0.0)`;

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      if (this.hasPreluActivationWeights) {
        activationSnippet = `vec4 activation(vec4 a, ivec3 outCoord) {
                  vec4 b = getPreluActivationWeightsAtOutCoords(outCoord);
                  ${this.activation}
                }`;
      } else {
        activationSnippet = `
                vec4 activation(vec4 a, ivec3 outCoord) {
                  ${this.activation}
                }`;
      }

      applyActivationSnippet = 'value = activation(value, outCoord);';
    }

    const addBiasSnippet =
        this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';

    const userCode = `
      ${activationSnippet}
      int dimAOuter = ${this.aShape[1]};
      int dimInner = ${this.aShape[2]};
      int dimBOuter = ${bShape[2]};
      int batch;

      ${makeMatMulPackedVec4Source([
      vecSize, this.workPerThread, 1
    ])}

      vec4 mm_readA(int row, int col) {
        return ${sampleA};
      }

      vec4 mm_readB(int row, int col) {
        return ${sampleB};
      }

      void mm_write(int row, int col, vec4 value) {
        if (row < dimAOuter && col * 4 < dimBOuter)
        {
          ivec3 outCoord = ivec3(batch, row, col * 4);
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutput(outCoord[0], outCoord[1], outCoord[2], value);
        }
      }

      void main() {
        batch = int(gl_GlobalInvocationID.z);
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    `;
    return userCode;
  }
}
