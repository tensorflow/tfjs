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

import {WebGPUProgram} from './webgpu_program';

export class MatMulProgram implements WebGPUProgram {
  outputShape: number[];
  userCode: string;
  dispatch: [number, number, number];
  variableNames = ['A', 'B', 'Dimensions'];
  tileSize = 8;

  constructor(outputShape: [number, number, number]) {
    this.outputShape = outputShape;
    this.dispatch = [
      Math.ceil(outputShape[1] / this.tileSize),
      Math.ceil(outputShape[2] / this.tileSize), 1
    ];

    this.userCode = `
      shared float Asub[TileSize][TileSize];
      shared float Bsub[TileSize][TileSize];

      void main() {
        // M is A outer, N is shared, K is B outer
        uint M = Dimensions[0], N = Dimensions[1], 
          K = Dimensions[2], batch = Dimensions[3];
        uint row = gl_LocalInvocationID.x; // Local row ID (max: TileSize)
        uint col = gl_LocalInvocationID.y; // Local col ID (max: TileSize)
        uint globalRow = TileSize*gl_WorkGroupID.x + row; // Row ID of C (0..M)
        uint globalCol = TileSize*gl_WorkGroupID.y + col; // Col ID of C (0..N)

        float acc = 0.0;

        uint numTiles = (N - 1)/TileSize + 1;

        for (uint t=0; t<numTiles; t++) {
          // Load one tile of A and B into local memory
          uint tiledRow = TileSize*t + row;
          uint tiledCol = TileSize*t + col;
          Asub[row][col] = A[globalRow*N + tiledCol];
          Bsub[row][col] = B[tiledRow*K + globalCol];

          // Synchronise to make sure the tile is loaded
          barrier();

          for (uint k=0; k<TileSize; k++) {
            acc += Asub[row][k] * Bsub[k][col];
          }

          // Synchronise before loading the next tile
          barrier();
        }

        if(globalCol < K && globalRow < M) {
          setOutput(globalRow*K + globalCol, acc);
        }
      }
    `;
  }
}