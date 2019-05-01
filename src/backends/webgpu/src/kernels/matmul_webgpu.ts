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

import {WebGPUProgram} from './webgpu_program';

export const matMulHeader = `
  float mm_readA(uint row, uint col);
  float mm_readB(uint row, uint col);
  void mm_write(uint row, uint col, float value);
  void mm_matMul(uint dimAOuter, uint dimInner, uint dimBOuter);`;

export function makeMatMulSource(): string {
  return `
    ${matMulHeader}

    const uint TileSide = TileSize.x;  // TileSize.x == TileSize.y
    shared float mm_Asub[TileSide][TileSide];
    shared float mm_Bsub[TileSide][TileSide];

    void mm_matMul(uint dimAOuter, uint dimInner, uint dimBOuter) {
        uint localRow = gl_LocalInvocationID.x;  // 0..TileSide
        uint localCol = gl_LocalInvocationID.y;  // 0..TileSide
        uint globalRow = TileSize.x * gl_WorkGroupID.x + localRow;  // AOuter
        uint globalCol = TileSize.x * gl_WorkGroupID.y + localCol;  // Inner

        float acc = 0.0;

        uint numTiles = (dimInner - 1) / TileSize.x + 1;

        for (uint t = 0; t < numTiles; t++) {
          // Load one tile of A and B into local memory
          uint tiledACol = TileSize.x * t + localCol;
          uint tiledBRow = TileSize.x * t + localRow;
          mm_Asub[localRow][localCol] = mm_readA(globalRow, tiledACol);
          mm_Bsub[localRow][localCol] = mm_readB(tiledBRow, globalCol);

          // Synchronise to make sure the tile is loaded
          barrier();

          for (uint k = 0; k < TileSize.x; k++) {
            acc += mm_Asub[localRow][k] * mm_Bsub[k][localCol];
          }

          // Synchronise before loading the next tile
          barrier();
        }

        if (globalCol < dimBOuter && globalRow < dimAOuter) {
          mm_write(globalRow, globalCol, acc);
        }
      }
  `;
}

export class MatMulProgram implements WebGPUProgram {
  outputShape: number[];
  userCode: string;
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  uniforms = 'uint dimAOuter, dimInner, dimBOuter, batch;';
  tileSize: [number, number, number] = [16, 16, 1];  // Must be square.

  constructor(outputShape: [number, number, number]) {
    this.outputShape = outputShape;
    const dispatchLayout = {x: [1], y: [2], z: [0]};
    this.dispatch =
        computeDispatch(dispatchLayout, this.outputShape, this.tileSize);

    this.userCode = `
      ${makeMatMulSource()}

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
