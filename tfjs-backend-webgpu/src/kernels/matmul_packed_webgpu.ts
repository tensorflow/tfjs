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

import {computeDispatch} from '../webgpu_util';

import {matMulHeader} from './matmul_webgpu';
import {WebGPUProgram} from './webgpu_program';

export function makeMatMulPackedSource(workPerThread: number): string {
  return `
    ${matMulHeader}

    const int WorkGroupSize = gl_WorkGroupSize.x;  // .x == .y
    const int WorkPerThread = ${workPerThread};
    const int MatTileSize = WorkGroupSize * WorkPerThread;

    shared float mm_Asub[MatTileSize][MatTileSize];
    shared float mm_Bsub[MatTileSize][MatTileSize];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      // These are 0..MatTileSize, in increments of WorkPerThread.
      int tileRow = gl_LocalInvocationID.y * WorkPerThread;
      int tileCol = gl_LocalInvocationID.x * WorkPerThread;

      // These are 0..AOuter, in increments of WorkPerThread.
      int globalRow = gl_GlobalInvocationID.y * WorkPerThread;
      int globalCol = gl_GlobalInvocationID.x * WorkPerThread;

      int numTiles = (dimInner - 1) / MatTileSize + 1;

      float acc[WorkPerThread][WorkPerThread];
      float ACached;
      float BCached[WorkPerThread];

      // Without this initialization strange values show up in acc.
      for (int innerRow = 0; innerRow < WorkPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < WorkPerThread; innerCol++) {
          acc[innerRow][innerCol] = 0.0;
        }
      }

      // Loop over shared dimension.
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A and B into local memory.
        for (int innerRow = 0; innerRow < WorkPerThread; innerRow++) {
          for (int innerCol = 0; innerCol < WorkPerThread; innerCol++) {
            int inputRow = tileRow + innerRow;
            int inputCol = tileCol + innerCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                t * MatTileSize + tileCol + innerCol);
            mm_Bsub[inputRow][inputCol] = mm_readB(
                t * MatTileSize + tileRow + innerRow,
                globalCol + innerCol);
          }
        }

        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < MatTileSize; k++) {
          for (int inner = 0; inner < WorkPerThread; inner++) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
          }

          for (int innerRow = 0; innerRow < WorkPerThread; innerRow++) {
            ACached = mm_Asub[tileRow + innerRow][k];
            for (int innerCol = 0; innerCol < WorkPerThread; innerCol++) {
              acc[innerRow][innerCol] += ACached * BCached[innerCol];
            }
          }
        }

        barrier();
      }

      for (int innerRow = 0; innerRow < WorkPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < WorkPerThread; innerCol++) {
          int globalFlatIndex =
            (globalRow + innerRow) * dimBOuter + (globalCol + innerCol);

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
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number] = [16, 16, 1];

  constructor(outputShape: [number, number, number], workPerThread: number) {
    this.outputShape = outputShape;
    this.workPerThread = workPerThread;

    this.dispatchLayout = {x: [1], y: [2], z: [0]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [workPerThread, workPerThread, 1]);

    // Consider compiling a different version of the shader that doesn't care
    // about boundary conditions when loading from Asub / Bsub when tiles fit
    // neatly inside of output. May slightly improve performance.
    this.userCode = `
      int dimAOuter = aShape[1];
      int dimInner = aShape[2];
      int dimBOuter = bShape[2];

      ${makeMatMulPackedSource(workPerThread)}

      float mm_readA(int row, int col) {
        if (row < dimAOuter && col < dimInner) {
          return A[row * dimInner + col];
        } else {
          return 0.0;
        }
      }

      float mm_readB(int row, int col) {
        if (row < dimInner && col < dimBOuter) {
          return B[row * dimBOuter + col];
        } else {
          return 0.0;
        }
      }

      void mm_write(int row, int col, float value) {
        setOutput(row * dimBOuter + col, value);
      }

      void main() {
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    `;
  }
}
