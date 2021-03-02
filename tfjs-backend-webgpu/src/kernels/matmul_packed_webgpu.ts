/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

import {util} from '@tensorflow/tfjs-core';

import {computeDispatch, computeWorkGroupSizeForMatMul, tilesFitEvenlyIntoShape} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export function makeMatMulPackedSource(workPerThread: number[]): string {
  return `
    float mm_readA(int row, int col);
    float mm_readB(int row, int col);
    void mm_write(int row, int col, float value);
    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter);

    const int RowPerThread = ${workPerThread[1]};
    const int ColPerThread = ${workPerThread[0]};
    const int TileAOuter = int(gl_WorkGroupSize.y) * RowPerThread;
    const int TileBOuter = int(gl_WorkGroupSize.x) * ColPerThread;
    const int TileInner = TileAOuter > TileBOuter ? TileAOuter : TileBOuter;

    shared float mm_Asub[TileAOuter][TileInner];
    shared float mm_Bsub[TileInner][TileBOuter];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
      int tileCol = int(gl_LocalInvocationID.x) * ColPerThread;

      int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
      int globalCol = int(gl_GlobalInvocationID.x) * ColPerThread;

      int numTiles = (dimInner - 1) / TileInner + 1;

      float acc[RowPerThread][ColPerThread];
      float ACached;
      float BCached[ColPerThread];

      // Without this initialization strange values show up in acc.
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
          acc[innerRow][innerCol] = 0.0;
        }
      }

      const int ColPerThreadA = TileInner / int(gl_WorkGroupSize.x);
      int tileColA = int(gl_LocalInvocationID.x) * ColPerThreadA;
      const int RowPerThreadB = TileInner / int(gl_WorkGroupSize.y);
      int tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;

      // Loop over shared dimension.
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          for (int innerCol = 0; innerCol < ColPerThreadA; innerCol++) {
            int inputRow = tileRow + innerRow;
            int inputCol = tileColA + innerCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                t * TileInner + inputCol);
          }
        }
        // Load one tile of B into local memory.
        for (int innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
          for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
            int inputRow = tileRowB + innerRow;
            int inputCol = tileCol + innerCol;

            mm_Bsub[inputRow][inputCol] = mm_readB(
              t * TileInner + inputRow,
              globalCol + innerCol);;
          }
        }

        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < TileInner; k++) {
          for (int inner = 0; inner < ColPerThread; inner++) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
          }

          for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
            ACached = mm_Asub[tileRow + innerRow][k];
            for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
              acc[innerRow][innerCol] += ACached * BCached[innerCol];
            }
          }
        }

        barrier();
      }

      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {

          if ((globalCol + innerCol) < dimBOuter &&
              (globalRow + innerRow) < dimAOuter) {
            mm_write(globalRow + innerRow,
                     globalCol + innerCol,
                     acc[innerRow][innerCol]);
          }
        }
      }
    }
  `;
}

export class MatMulPackedProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number] = [16, 16, 1];
  aShape: [number, number, number];
  transposeA: boolean;
  transposeB: boolean;
  addBias: boolean;
  activation: string;
  hasPreluActivationWeights: boolean;

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      workPerThread: number, transposeA = false, transposeB = false,
      addBias = false, activation: string = null,
      hasPreluActivationWeights = false) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    const dimInner = transposeA ? aShape[1] : aShape[2];
    this.workGroupSize =
        computeWorkGroupSizeForMatMul(outputShape[1], dimInner, outputShape[2]);
    // TODO: Consider to use a seperate algorithm to optimize it when the output
    // is a vector.
    if (outputShape[1] === 1 || outputShape[2] === 1) {
      workPerThread = 1;
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [workPerThread, workPerThread, 1]);
    // If dispaching number is one, it means only one work group is running.
    // For modern GPUs, it supports multiple work groups running in parallel.
    // So there may be some idle hardware threads.
    // In this case, we prefer to reduce the work per thread and improve the
    // thread utilization
    if (util.arraysEqual(this.dispatch, [1, 1, 1])) {
      workPerThread = 1;
      this.dispatch = computeDispatch(
          this.dispatchLayout, this.outputShape, this.workGroupSize,
          [workPerThread, workPerThread, 1]);
    }

    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.workPerThread = workPerThread;
    this.aShape = aShape;
    this.transposeA = transposeA;
    this.transposeB = transposeB;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.shaderKey = `matMulPacked_${this.workPerThread}_${transposeA}_${
        transposeB}_${activation}`;
  }

  getUserCode(): string {
    const dimInner = this.transposeA ? this.aShape[1] : this.aShape[2];
    const dimBOuter = this.outputShape[2];
    const bShape = this.transposeB ?
        [this.outputShape[0], dimBOuter, dimInner] :
        [this.outputShape[0], dimInner, dimBOuter];

    const tileAOuter = this.workGroupSize[1] * this.workPerThread;
    const tileBOuter = this.workGroupSize[0] * this.workPerThread;
    const tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
    util.assert(
        tileInner % this.workGroupSize[0] === 0 &&
            tileInner % this.workGroupSize[1] === 0,
        () => `tileInner must be multiple of workgroupsize.x ` +
            `and workgroupsize.y`);
    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];
    const fitA = tilesFitEvenlyIntoShape(tileSizeA, this.aShape.slice(1));
    const batchASize = this.aShape[1] * this.aShape[2];
    const batchBSize = bShape[1] * bShape[2];
    let sampleA;

    if (this.transposeA === false) {
      sampleA = fitA ?
          `A[batch * ${batchASize} + row * dimInner + col]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            A[batch * ${batchASize} + row * dimInner + col] : 0`;
    } else {
      sampleA = fitA ?
          `A[batch * ${batchASize} + col * dimAOuter + row]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            A[batch* ${batchASize} + col * dimAOuter + row] : 0`;
    }

    const fitB = tilesFitEvenlyIntoShape(tileSizeB, bShape.slice(1));
    let sampleB;
    if (this.transposeB === false) {
      sampleB = fitB ?
          `B[batch * ${batchBSize} + row * dimBOuter + col]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B[batch * ${batchBSize} + row * dimBOuter + col] : 0`;
    } else {
      sampleB = fitB ?
          `B[batch * ${batchBSize} + col * dimInner + row]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B[batch * ${batchBSize} + col * dimInner + row] : 0`;
    }

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      if (this.hasPreluActivationWeights) {
        activationSnippet = `float activation(float a, ivec3 outCoord) {
              float b = getPreluActivationWeightsAtOutCoords(outCoord);
              ${this.activation}
            }`;
      } else {
        activationSnippet = `
              float activation(float a, ivec3 outCoord) {
                ${this.activation}
              }
            `;
      }

      applyActivationSnippet = 'value = activation(value, outCoord);';
    }

    const addBiasSnippet =
        this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';

    const userCode = `
      ${activationSnippet}

      int dimAOuter = ${
        this.transposeA === true ? `${this.aShape[2]}` : `${this.aShape[1]}`};
      int dimInner = ${
        this.transposeA === true ? `${this.aShape[1]}` : `${this.aShape[2]}`};
      int dimBOuter = ${
        this.transposeB === true ? `${bShape[1]}` : `${bShape[2]}`};

      int batch;

      ${makeMatMulPackedSource([
      this.workPerThread, this.workPerThread, 1
    ])}
      float mm_readA(int row, int col) {
        return ${sampleA};
      }
      float mm_readB(int row, int col) {
        return ${sampleB};
      }
      void mm_write(int row, int col, float value) {
        ivec3 outCoord = ivec3(batch, row, col);
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(batch, row, col, value);
      }
      void main() {
        batch = int(gl_GlobalInvocationID.z);
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    `;
    return userCode;
  }
}
