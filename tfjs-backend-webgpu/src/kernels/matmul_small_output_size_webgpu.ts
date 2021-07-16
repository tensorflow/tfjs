/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {backend_util, TensorInfo, util} from '@tensorflow/tfjs-core';
import {WebGPUProgram} from './webgpu_program';

export function makeMatMulSmallOutputSizeWidthSource(
    aOuterSmall: boolean): string {
  return `
  float mm_readA(int row, int col);
  float mm_readB(int row, int col);
  void mm_write(int row, int col, float value);

  const int TileAOuter = int(gl_WorkGroupSize.x);
  const int TileBOuter = int(gl_WorkGroupSize.x);
  const int TileInner = int(gl_WorkGroupSize.y - gl_WorkGroupSize.x);

  shared float mm_Asub1[TileAOuter][TileInner];
  shared float mm_Bsub1[TileInner][TileBOuter];
  shared float mm_Asub2[TileAOuter][TileInner];
  shared float mm_Bsub2[TileInner][TileBOuter];

  void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
    int tileRowA = int(gl_LocalInvocationID.x);
    int tileColA = int(gl_LocalInvocationID.y);
    int tileRowB = int(gl_LocalInvocationID.y);
    int tileColB = int(gl_LocalInvocationID.x);

    int ${aOuterSmall ? 'globalColB' : 'globalRowA'} =
        int(gl_GlobalInvocationID.x);

    int numTiles = (dimInner - 1) / TileInner + 1;
    float acc = 0.0;

    for (int t = 0; t < numTiles; t++) {
      if (t == 0) {
        if (tileColA < TileInner) {
          // Load one tile of A and B into local memory.
          mm_Asub1[tileRowA][tileColA] =
              mm_readA(${aOuterSmall ? 'tileRowA' : 'globalRowA'}, tileColA);
          mm_Bsub1[tileRowB][tileColB] =
              mm_readB(tileRowB, ${aOuterSmall ? 'globalColB' : 'tileColB'});
        }
      } else {
        if (tileColA < TileInner) {
          // Load one tile of A and B into local memory.
          mm_Asub1[tileRowA][tileColA] =
              mm_readA(${aOuterSmall ? 'tileRowA' : 'globalRowA'},
                  t * TileInner + tileColA);
          mm_Bsub1[tileRowB][tileColB] =
              mm_readB(t * TileInner + tileRowB,
                  ${aOuterSmall ? 'globalColB' : 'tileColB'});
        } else {
          // Compute acc values for a single thread.
          for (int k = 0; k < TileInner; k++) {
            acc += mm_Asub2[tileRowA][k] * mm_Bsub2[k][tileColA-TileInner];
          }
        }
      }

      barrier();
      if (t != 0) {
        t++;
      }

      if (t < numTiles) {
        if (tileColA < TileInner) {
          // Load one tile of A and B into local memory.
          mm_Asub2[tileRowA][tileColA] =
              mm_readA(${aOuterSmall ? 'tileRowA' : 'globalRowA'},
                  t * TileInner + tileColA);
          mm_Bsub2[tileRowB][tileColB] =
              mm_readB(t * TileInner + tileRowB,
                  ${aOuterSmall ? 'globalColB' : 'tileColB'});
        } else {
          // Compute acc values for a single thread.
          for (int k = 0; k < TileInner; k++) {
            acc += mm_Asub1[tileRowA][k] * mm_Bsub1[k][tileColA-TileInner];
          }
        }
      }

      barrier();
    }

    if (tileColA >= TileInner) {
      mm_write(${aOuterSmall ? 'tileRowA' : 'globalRowA'},
          ${aOuterSmall ? 'int(gl_WorkGroupSize.x * gl_WorkGroupID.x) + '
              : ''}tileColA - TileInner, acc);
    }
  }
  `;
}

export class MatMulSmallOutputSizeWidthProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number] = [8, 24, 1];
  aOuterSmall: boolean;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;

  constructor(
      aShape: [number, number, number], bShape: [number, number, number],
      outputShape: [number, number, number], bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    util.assert(aShape[1] <= 16 || bShape[2] <= 16,
      () => 'This program can be only used when A width is small.');
    this.outputShape = outputShape;

    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    if (aShape[1] <= 16) {
      this.workGroupSize = aShape[1] <= 8 ? [8, 24, 1] : [16, 32, 1];
      this.dispatch = [Math.ceil(outputShape[2] / this.workGroupSize[0]), 1, 1];
    } else if (bShape[2] <= 16) {
      this.workGroupSize = bShape[2] <= 8 ? [8, 24, 1] : [16, 32, 1];
      this.dispatch = [Math.ceil(outputShape[1] / this.workGroupSize[0]), 1, 1];
    }

    const addBias = bias != null;
    if (addBias) {
      this.variableNames.push('bias');
    }

    const hasPreluActivationWeights = preluActivationWeights != null;
    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.aOuterSmall = aShape[1] <= 16;
    if (aShape[1] <= 8) {
      this.shaderKey = `matMulSmallOutputSize_A_8_${this.activation}`;
    } else if (aShape[1] <= 16){
      this.shaderKey = `matMulSmallOutputSize_A_16_${this.activation}`;
    } else if (bShape[2] <= 8) {
      this.shaderKey = `matMulSmallOutputSize_B_8_${this.activation}`;
    } else if (bShape[2] <= 16){
      this.shaderKey = `matMulSmallOutputSize_B_16_${this.activation}`;
    }
  }

  getUserCode(): string {
    let sampleA;

      sampleA = `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            A[batch * batchASize + row * dimInner + col] : 0`;

    let sampleB;
      sampleB = `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B[batch * batchBSize + row * dimBOuter + col] : 0`;

    const userCode = `

      int dimAOuter = aShape[1];
      int dimInner = aShape[2];
      int dimBOuter = bShape[2];

      int batch;
      ${makeMatMulSmallOutputSizeWidthSource(this.aOuterSmall)}
      float mm_readA(int row, int col) {
        int batchASize = aShape[1] * aShape[2];
        return ${sampleA};
      }
      float mm_readB(int row, int col) {
        int batchBSize = bShape[1] * bShape[2];
        return ${sampleB};
      }
      void mm_write(int row, int col, float value) {
        ivec3 outCoord = ivec3(batch, row, col);
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
