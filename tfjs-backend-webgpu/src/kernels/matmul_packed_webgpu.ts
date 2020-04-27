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
import {computeDispatch, tilesFitEvenlyIntoShape} from '../webgpu_util';

import {matMulHeader} from './matmul_webgpu';
import {WebGPUProgram} from './webgpu_program';

export function makeMatMulPackedSource(workPerThread: number[]): string {
  return `
    ${matMulHeader}

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
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number] = [16, 16, 1];

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      workPerThread: number, transposeA = false, transposeB = false) {
    const dimInner = transposeA ? aShape[1] : aShape[2];
    const dimBOuter = outputShape[2];
    const bShape = transposeB ? [outputShape[0], dimBOuter, dimInner] :
                                [outputShape[0], dimInner, dimBOuter];
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [workPerThread, workPerThread, 1]);
    // If dispaching number is one, it means only one work group is running.
    // For modern GPUs, it supports multiple work groups running in parallel.
    // So there may be some idle hardware threads.
    // In this case, we prefer to reduce the work per thread and improve the
    // thread utilization
    if (this.dispatch === [1, 1, 1]) {
      workPerThread = 1;
      this.dispatch = computeDispatch(
          this.dispatchLayout, this.outputShape, this.workGroupSize,
          [workPerThread, workPerThread, 1]);
    }
    this.workPerThread = workPerThread;
    const tileAOuter = this.workGroupSize[1] * workPerThread;
    const tileBOuter = this.workGroupSize[0] * workPerThread;
    const tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
    util.assert(tileInner % this.workGroupSize[0] === 0 &&
                tileInner % this.workGroupSize[1] === 0,
                () => 'tileInner must be multiple of workgroupsize.x and workgroupsize.y');
    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];
    const fitA = tilesFitEvenlyIntoShape(tileSizeA, aShape.slice(1));
    let sampleA;
    if (transposeA === false) {
      sampleA = fitA ?
          `A[row * dimInner + col]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            A[row * dimInner + col] : 0`;
    } else {
      sampleA = fitA ?
          `A[col * dimAOuter + row]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            A[col * dimAOuter + row] : 0`;
    }

    const fitB = tilesFitEvenlyIntoShape(tileSizeB, bShape.slice(1));
    let sampleB;
    if (transposeB === false) {
      sampleB = fitB ?
          `B[row * dimBOuter + col]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B[row * dimBOuter + col] : 0`;
    } else {
      sampleB = fitB ?
          `B[col * dimInner + row]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B[col * dimInner + row] : 0`;
    }

    this.userCode = `
      int dimAOuter = ${transposeA === true ? `aShape[2]` : `aShape[1]`};
      int dimInner = ${transposeA === true ? `aShape[1]` : `aShape[2]`};
      int dimBOuter = ${transposeB === true ? `bShape[1]` : `bShape[2]`};

      ${makeMatMulPackedSource([
      workPerThread, workPerThread, 1
    ])}
      float mm_readA(int row, int col) {
        return ${sampleA};
      }

      float mm_readB(int row, int col) {
        return ${sampleB};
      }

      void mm_write(int row, int col, float value) {
        setOutput(row * dimBOuter + col, value);
      }

      void main() {
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    `;
    this.shaderKey = `matmulpacked${this.workPerThread}${fitA}${fitB}${
        transposeA}${transposeB}`;
  }
}
