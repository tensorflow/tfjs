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

import {computeDispatch, tilesFitEvenlyIntoShape} from '../webgpu_util';

import {matMulHeader} from './matmul_webgpu';
import {WebGPUProgram} from './webgpu_program';

function fillTileB(workGroupSize: number[]): string {
  let result = '';
  if (workGroupSize[0] > workGroupSize[1]) {
    const count = workGroupSize[0] / workGroupSize[1];
    result = `
            int inputBRow = ${count} * inputRow;`;
    for (let i = 0; i < count; i++) {
      result += `
            mm_Bsub[inputBRow + ${i}][inputCol] = mm_readB(
              t * Tile_K + inputBRow + ${i},
              globalCol + innerCol);`;
    }
  } else {
    result = `
            mm_Bsub[inputRow][inputCol] = mm_readB(
              t * Tile_K + inputRow,
              globalCol + innerCol);`;
  }
  return result;
}

export function makeMatMulPackedSource(
    workPerThread: number, workGroupSize: number[]): string {
  return `
    ${matMulHeader}

    const int WorkPerThread = ${workPerThread};
    const int Tile_M = int(gl_WorkGroupSize.y) * WorkPerThread;
    const int Tile_N = int(gl_WorkGroupSize.x) * WorkPerThread;
    const int Tile_K = Tile_N;

    shared float mm_Asub[Tile_M][Tile_K];
    shared float mm_Bsub[Tile_K][Tile_N];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      // These are 0..MatTileSize, in increments of WorkPerThread.
      int tileRow = int(gl_LocalInvocationID.y) * WorkPerThread;
      int tileCol = int(gl_LocalInvocationID.x) * WorkPerThread;

      // These are 0..AOuter, in increments of WorkPerThread.
      int globalRow = int(gl_GlobalInvocationID.y) * WorkPerThread;
      int globalCol = int(gl_GlobalInvocationID.x) * WorkPerThread;

      int numTiles = (dimInner - 1) / Tile_K + 1;

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
                t * Tile_K + tileCol + innerCol);
            ${fillTileB(workGroupSize)};
          }
        }

        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < Tile_K; k++) {
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

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      workPerThread: number) {
    const bShape = [outputShape[0], aShape[2], outputShape[2]];
    this.outputShape = outputShape;
    this.workPerThread = workPerThread;

    const sampleA = tilesFitEvenlyIntoShape(
                        this.workGroupSize.slice(0, 2), aShape.slice(1)) ?
        `A[row * dimInner + col]` :
        `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
          A[row * dimInner + col] : 0`;
    const sampleB = tilesFitEvenlyIntoShape(
                        this.workGroupSize.slice(0, 2), bShape.slice(1)) ?
        `B[row * dimBOuter + col]` :
        `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
          B[row * dimBOuter + col] : 0`;

    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [workPerThread, workPerThread, 1]);

    this.userCode = `
      int dimAOuter = aShape[1];
      int dimInner = aShape[2];
      int dimBOuter = bShape[2];
      ${makeMatMulPackedSource(workPerThread, this.workGroupSize)}
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
  }
}
