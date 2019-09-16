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

export class MatMulPackedV2Program implements WebGPUProgram {
  outputShape: number[];
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number] = [16, 4, 1];
  isVec4 = true;

  constructor(outputShape: [number, number, number]) {
    this.outputShape = outputShape;
    const workPerThreadY = 8;
    const workPerThreadX = 4;

    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [workPerThreadX, workPerThreadY, 1]);

    this.userCode = `
      const int TILE_M = ${workPerThreadY * this.workGroupSize[1]}; // 32
      const int TILE_N = ${workPerThreadX * this.workGroupSize[0]}; // 64
      const int VEC_SIZE = 4;
      const int TILE_K = VEC_SIZE * ${this.workGroupSize[0]}; // 64
      const int ROWS_PER_WI = ${workPerThreadY};  // 8

      shared vec4 atile[TILE_M * TILE_K / VEC_SIZE];

      int dimAOuter = aShape[1];
      int dimInner = aShape[2] / VEC_SIZE;
      int dimBOuter = bShape[2] / VEC_SIZE;

      // Consider compiling a different version of the shader that doesn't care
      // about boundary conditions. May slightly improve performance.
      vec4 mm_readA(int row, int col) {
        if (row < dimAOuter && col < dimInner) {
          vec4 result = A[row * dimInner + col];
          return result;
        } else {
          return vec4(0.0, 0.0, 0.0, 0.0);
        }
      }

      vec4 mm_readB(int row, int col) {
        if (row < aShape[2] && col < dimBOuter) {
          vec4 result = B[row * dimBOuter + col];
          return result;
        } else {
          return vec4(0.0, 0.0, 0.0, 0.0);
        }
      }

      void mm_write(int row, int col, vec4 value) {
        if (row < dimAOuter && col < dimBOuter)
        {
          result[row * dimBOuter + col] = value;
        }
      }

      void main() {
        int group_x = int(gl_WorkGroupID.x);
        int group_y = int(gl_WorkGroupID.y);
        int local_x = int(gl_LocalInvocationID.x);
        int local_y = int(gl_LocalInvocationID.y);

        // Result ctile is M rows x N columns
        // M = 32, we have 4 rows of work-items, so we need 32/4 8 results down
        // N = 64, we have 16 columns of work-items, so we need 64/16 = 4
        // results across = 1 float4 across

        vec4 dot00,dot01,dot02,dot03, dot04, dot05, dot06, dot07;

        // Src0 is used to load atile.
        // It starts at the left side of src0 and walks across.
        // atile is M rows x K columns.
        int globalRow = ( group_y * TILE_M ) + ROWS_PER_WI * local_y;
        int globalColA = local_x;

        // Src1 is directly used as btile.
        // It starts at the top of src1 and walks down.
        // btile is K rows x N columns.
        // K = 64, we'll process four rows at a time
        // N = 64, we have 16 columns of work-items, so we need 64/16 = 4 floats
        // across = 1 float4 across
        int globalCol0 = local_x + ( group_x * ( TILE_N / VEC_SIZE ) );
        int rowB0 = 0;

        int slm = local_y * ( ROWS_PER_WI * TILE_K / VEC_SIZE );

        // Walk ACROSS src0 and DOWN src1:
        int w = 0;
        do{
          // We want to load atile, which is M rows x K columns
          // M = 32, and we have 4 rows of work-items, so each work-item must
          // load 32/4 = 8 rows.
          // K = 64, and we have 16 columns of work-items, so each work-item
          // must load 64/16 = 4 columns = 1 float4.
          atile[slm + local_x + 0 * TILE_K / VEC_SIZE] =
              mm_readA(globalRow, globalColA);
          atile[slm + local_x + 1 * TILE_K / VEC_SIZE] =
              mm_readA(globalRow + 1, globalColA);
          atile[slm + local_x + 2 * TILE_K / VEC_SIZE] =
              mm_readA(globalRow + 2, globalColA);
          atile[slm + local_x + 3 * TILE_K / VEC_SIZE] =
              mm_readA(globalRow + 3, globalColA);
          atile[slm + local_x + 4 * TILE_K / VEC_SIZE] =
              mm_readA(globalRow + 4, globalColA);
          atile[slm + local_x + 5 * TILE_K / VEC_SIZE] =
              mm_readA(globalRow + 5, globalColA);
          atile[slm + local_x + 6 * TILE_K / VEC_SIZE] =
              mm_readA(globalRow + 6, globalColA);
          atile[slm + local_x + 7 * TILE_K / VEC_SIZE] =
              mm_readA(globalRow + 7, globalColA);

          globalColA += TILE_K / VEC_SIZE;

          barrier();

          int i = 0;
          do{
            // We get better performance by loading btile first.
            vec4 brow00 = mm_readB(rowB0, globalCol0); rowB0++;
            vec4 brow01 = mm_readB(rowB0, globalCol0); rowB0++;
            vec4 brow02 = mm_readB(rowB0, globalCol0); rowB0++;
            vec4 brow03 = mm_readB(rowB0, globalCol0); rowB0++;

            vec4 a0 = atile[slm + i + 0 * TILE_K / VEC_SIZE ];
            dot00 = brow00*a0.x + dot00;
            dot00 = brow01*a0.y + dot00;
            dot00 = brow02*a0.z + dot00;
            dot00 = brow03*a0.w + dot00;

            vec4 a1 = atile[slm + i + 1 * TILE_K / VEC_SIZE ];
            dot01 = brow00*a1.x + dot01;
            dot01 = brow01*a1.y + dot01;
            dot01 = brow02*a1.z + dot01;
            dot01 = brow03*a1.w + dot01;

            vec4 a2 = atile[slm + i + 2 * TILE_K / VEC_SIZE ];
            dot02 = brow00*a2.x + dot02;
            dot02 = brow01*a2.y + dot02;
            dot02 = brow02*a2.z + dot02;
            dot02 = brow03*a2.w + dot02;

            vec4 a3 = atile[slm + i + 3 * TILE_K / VEC_SIZE ];
            dot03 = brow00*a3.x + dot03;
            dot03 = brow01*a3.y + dot03;
            dot03 = brow02*a3.z + dot03;
            dot03 = brow03*a3.w + dot03;

            vec4 a4 = atile[slm + i + 4 * TILE_K / VEC_SIZE ];
            dot04 = brow00*a4.x + dot04;
            dot04 = brow01*a4.y + dot04;
            dot04 = brow02*a4.z + dot04;
            dot04 = brow03*a4.w + dot04;

            vec4 a5 = atile[slm + i + 5 * TILE_K / VEC_SIZE ];
            dot05 = brow00*a5.x + dot05;
            dot05 = brow01*a5.y + dot05;
            dot05 = brow02*a5.z + dot05;
            dot05 = brow03*a5.w + dot05;

            vec4 a6 = atile[slm + i + 6 * TILE_K / VEC_SIZE ];
            dot06 = brow00*a6.x + dot06;
            dot06 = brow01*a6.y + dot06;
            dot06 = brow02*a6.z + dot06;
            dot06 = brow03*a6.w + dot06;

            vec4 a7 = atile[slm + i + 7 * TILE_K / VEC_SIZE ];
            dot07 = brow00*a7.x + dot07;
            dot07 = brow01*a7.y + dot07;
            dot07 = brow02*a7.z + dot07;
            dot07 = brow03*a7.w + dot07;

            i++;
          }
          while( i < TILE_K / VEC_SIZE );

          barrier();

          w += TILE_K / VEC_SIZE;
        }
        while( w < dimInner );

        mm_write(globalRow, globalCol0, dot00);
        mm_write(globalRow + 1, globalCol0, dot01);
        mm_write(globalRow + 2, globalCol0, dot02);
        mm_write(globalRow + 3, globalCol0, dot03);
        mm_write(globalRow + 4, globalCol0, dot04);
        mm_write(globalRow + 5, globalCol0, dot05);
        mm_write(globalRow + 6, globalCol0, dot06);
        mm_write(globalRow + 7, globalCol0, dot07);
      }
    `;
  }
}
