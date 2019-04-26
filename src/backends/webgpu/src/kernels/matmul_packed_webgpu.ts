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

export class MatMulPackedProgram implements WebGPUProgram {
  outputShape: number[];
  userCode: string;
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  uniforms = 'uint dimAOuter, dimInner, dimBOuter, batch;';
  tileSize: [number, number] = [32, 32];

  constructor(outputShape: [number, number, number], workPerThread: number) {
    this.outputShape = outputShape;
    this.workPerThread = workPerThread;
    this.dispatch = [
      Math.ceil(outputShape[1] / this.tileSize[0]),
      Math.ceil(outputShape[2] / this.tileSize[1]), 1
    ];

    // Consider compiling a different version of the shader that doesn't care
    // about boundary conditions when loading from Asub / Bsub when tiles fit
    // neatly inside of output. May slightly improve performance.
    this.userCode = `
      const uint WorkPerThread = ${workPerThread};
      shared float Asub[TileSize.x * WorkPerThread][TileSize.x * WorkPerThread];
      shared float Bsub[TileSize.x * WorkPerThread][TileSize.x * WorkPerThread];

      void main() {
        uint row = gl_LocalInvocationID.y; // 0..local_size_x
        uint col = gl_LocalInvocationID.x; // 0..local_size_y
        uint tileRow = row * WorkPerThread; // 0..TileSize, stride by local_size
        uint tileCol = col * WorkPerThread; // 0..TileSize
        
        // 0..AOuter, stride by tileSize
        uint globalRow = TileSize.x*gl_WorkGroupID.y + tileRow; 
        uint globalCol = TileSize.x*gl_WorkGroupID.x + tileCol;

        uint numTiles = (dimInner - 1) / TileSize.x + 1;

        float acc[WorkPerThread][WorkPerThread];
        float ACached;
        float BCached[WorkPerThread];

        // Without this initialization strange values show up in acc.
        for(uint innerRow=0; innerRow<WorkPerThread; innerRow++) {
          for(uint innerCol=0; innerCol<WorkPerThread; innerCol++) {
            acc[innerRow][innerCol] = 0.0;
          }
        }

        // Loop over shared dimension.
        for(uint t=0; t<numTiles; t++) { 
          // Load one tile of A and B into local memory.
          for(uint innerRow=0; innerRow<WorkPerThread; innerRow++) {
            for(uint innerCol=0; innerCol<WorkPerThread; innerCol++) {
              uint inputRow = tileRow + innerRow;
              uint inputCol = tileCol + innerCol;
              
              uint AColumnIndex = t * TileSize.x + tileCol + innerCol;
              uint AFlatIndex = 
                (globalRow + innerRow) * dimInner + AColumnIndex;

              if(AColumnIndex < dimInner && AFlatIndex < dimAOuter * dimInner) {
                Asub[inputRow][inputCol] = A[AFlatIndex];
              } else {
                Asub[inputRow][inputCol] = 0.0;
              }

              uint BRowIndex = t * TileSize.x + tileRow + innerRow;
              uint BFlatIndex = BRowIndex * dimBOuter + (globalCol + innerCol);

              if(BRowIndex < dimInner && BFlatIndex < dimInner * dimBOuter) {
                Bsub[inputRow][inputCol] = B[BFlatIndex];
              } else {
                Bsub[inputRow][inputCol] = 0.0; 
              }
            }
          }

          barrier();

          // Compute acc values for a single thread.
          for(uint k=0; k<TileSize.x; k++) {
            for(uint inner=0; inner<WorkPerThread; inner++) {
              BCached[inner] = Bsub[k][tileCol + inner];
            }
            
            for(uint innerRow=0; innerRow<WorkPerThread; innerRow++) {
              ACached = Asub[tileRow + innerRow][k];
              for(uint innerCol=0; innerCol<WorkPerThread; innerCol++) {
                acc[innerRow][innerCol] += ACached * BCached[innerCol];
              }
            }
          }

          barrier();
        }

        for (uint innerRow=0; innerRow<WorkPerThread; innerRow++) {
          for (uint innerCol=0; innerCol<WorkPerThread; innerCol++) {
            uint globalFlatIndex = 
              (globalRow + innerRow) * dimBOuter + (globalCol + innerCol);
            
            if((globalCol + innerCol) < dimBOuter && 
              (globalRow + innerRow) < dimAOuter) {
              setOutput(globalFlatIndex, acc[innerRow][innerCol]);
            }
          }
        }
      }
    `;
  }
}
