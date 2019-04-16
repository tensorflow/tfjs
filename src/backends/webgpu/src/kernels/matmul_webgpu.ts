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

  constructor(outputShape: [number, number, number]) {
    this.outputShape = outputShape;
    const tileSize = 2;
    this.dispatch = [
      Math.ceil(outputShape[1] / tileSize),
      Math.ceil(outputShape[2] / tileSize), 1
    ];

    this.userCode = `
      #version 450
      const uint TileSize = ${tileSize};
      layout (local_size_x = TileSize, local_size_y = TileSize, 
        local_size_z = 1) in;
      layout(std430, binding = 0) readonly buffer ssbA {
        float A[];
      };
      layout(std430, binding = 1) readonly buffer ssbB {
        float B[];
      };
      layout(std430, binding = 2) readonly buffer ssbDimensions {
        uint Dimensions[];
      };
      layout(std430, binding = 3) writeonly buffer ssbOut {
        float result[];
      };

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

        // Add 1 to N to ceil.
        uint numTiles = (N + 1)/TileSize;

        for (uint t=0; t<numTiles; t++) {
          // Load one tile of A and B into local memory
          uint tiledRow = TileSize*t + row;
          uint tiledCol = TileSize*t + col;
          Asub[row][col] = A[globalRow*N + tiledCol];
          Bsub[row][col] = B[tiledRow*K + globalCol];

          // Synchronise to make sure the tile is loaded
          // memoryBarrierShared();
          barrier();

          for (uint k=0; k<TileSize; k++) {
            acc += Asub[row][k] * Bsub[k][col];
          }

          // Synchronise before loading the next tile
          barrier();
        }

        if(globalCol < K && globalRow < M) {
          result[globalRow*K + globalCol] = acc;
        }
      }
    `;
  }
}