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

    const uint TileSide = TileSize.x;  // TileSize.x == TileSize.y
    const uint WorkPerThread = ${workPerThread};
    shared float mm_Asub[TileSide * WorkPerThread][TileSide * WorkPerThread];
    shared float mm_Bsub[TileSide * WorkPerThread][TileSide * WorkPerThread];

    void mm_matMul(uint dimAOuter, uint dimInner, uint dimBOuter) {
      uint row = gl_LocalInvocationID.y;   // 0..local_size_x
      uint col = gl_LocalInvocationID.x;   // 0..local_size_y
      uint tileRow = row * WorkPerThread;  // 0..TileSide, stride by local_size
      uint tileCol = col * WorkPerThread;  // 0..TileSide

      // 0..AOuter, stride by tileSize
      uint globalRow = TileSide * gl_WorkGroupID.y + tileRow;
      uint globalCol = TileSide * gl_WorkGroupID.x + tileCol;

      uint numTiles = (dimInner - 1) / TileSize.x + 1;

      float acc[WorkPerThread][WorkPerThread];
      float ACached;
      float BCached[WorkPerThread];

      // Without this initialization strange values show up in acc.
      for (uint innerRow = 0; innerRow < WorkPerThread; innerRow++) {
        for (uint innerCol = 0; innerCol < WorkPerThread; innerCol++) {
          acc[innerRow][innerCol] = 0.0;
        }
      }

      // Loop over shared dimension.
      for (uint t = 0; t < numTiles; t++) {
        // Load one tile of A and B into local memory.
        for (uint innerRow = 0; innerRow < WorkPerThread; innerRow++) {
          for (uint innerCol = 0; innerCol < WorkPerThread; innerCol++) {
            uint inputRow = tileRow + innerRow;
            uint inputCol = tileCol + innerCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                t * TileSize.x + tileCol + innerCol);
            mm_Bsub[inputRow][inputCol] = mm_readB(
                t * TileSize.x + tileRow + innerRow,
                globalCol + innerCol);
          }
        }

        barrier();

        // Compute acc values for a single thread.
        for (uint k = 0; k < TileSize.x; k++) {
          for (uint inner = 0; inner < WorkPerThread; inner++) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
          }

          for (uint innerRow = 0; innerRow < WorkPerThread; innerRow++) {
            ACached = mm_Asub[tileRow + innerRow][k];
            for (uint innerCol = 0; innerCol < WorkPerThread; innerCol++) {
              acc[innerRow][innerCol] += ACached * BCached[innerCol];
            }
          }
        }

        barrier();
      }

      for (uint innerRow = 0; innerRow < WorkPerThread; innerRow++) {
        for (uint innerCol = 0; innerCol < WorkPerThread; innerCol++) {
          uint globalFlatIndex =
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
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  uniforms = 'uint dimAOuter, dimInner, dimBOuter, batch;';
  tileSize: [number, number, number] = [16, 16, 1];

  constructor(outputShape: [number, number, number], workPerThread: number) {
    this.outputShape = outputShape;
    this.workPerThread = workPerThread;

    const dispatchLayout = {x: [1], y: [2], z: [0]};
    this.dispatch =
        computeDispatch(dispatchLayout, this.outputShape, this.tileSize);

    // Consider compiling a different version of the shader that doesn't care
    // about boundary conditions when loading from Asub / Bsub when tiles fit
    // neatly inside of output. May slightly improve performance.
    this.userCode = `
      ${makeMatMulPackedSource(workPerThread)}

      float mm_readA(uint row, uint col) {
        if (row < dimAOuter && col < dimInner) {
          return A[row * dimInner + col];
        } else {
          return 0.0;
        }
      }

      float mm_readB(uint row, uint col) {
        if (row < dimInner && col < dimBOuter) {
          return B[row * dimBOuter + col];
        } else {
          return 0.0;
        }
      }

      void mm_write(uint row, uint col, float value) {
        setOutput(row * dimBOuter + col, value);
      }

      void main() {
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    `;
  }
}
