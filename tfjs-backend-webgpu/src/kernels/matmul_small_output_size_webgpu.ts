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
import {getMainHeaderStringWgsl} from '../shader_preprocessor_wgsl';
import {mapActivationToShaderProgram} from './activation_util';
import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export function makeMatMulSmallOutputSizeSource(): string {
  return `
  float mm_readA(int row, int col);
  float mm_readB(int row, int col);
  void mm_write(int row, int col, float value);
  const int TileAOuter = int(gl_WorkGroupSize.y / 2);
  const int TileBOuter = int(gl_WorkGroupSize.x);
  const int TileInner = TileAOuter > TileBOuter ? TileAOuter : TileBOuter;

  shared float mm_Asub1[TileAOuter][TileInner];
  shared float mm_Bsub1[TileInner][TileBOuter];
  shared float mm_Asub2[TileAOuter][TileInner];
  shared float mm_Bsub2[TileInner][TileBOuter];

  // If the output size is small for matrix multiplication, avoid to use vec4
  // and handle some elements per thread to optimally utilize the ALU.
  // Introduces two shared memory buffers, some logical threads could handle
  // arithmetic operations and others handle IO operations between barrier api,
  // makes ALUs and load/store units work simultaneously, could improves
  // the performance.
  void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
    int tileRow = int(gl_LocalInvocationID.y);
    int tileCol = int(gl_LocalInvocationID.x);
    int globalRow = int(gl_GlobalInvocationID.y);
    int globalCol = int(gl_GlobalInvocationID.x);

    int numTiles = (dimInner - 1) / TileInner + 1;
    float acc = 0.0;

    int globalColA = tileCol;
    int globalRowB = tileRow;
    int tileColA = int(gl_LocalInvocationID.x);
    int tileRowB = int(gl_LocalInvocationID.y);
    for (int t = 0; t < numTiles; t++) {
      if (t == 0) {
        if (tileRow < TileAOuter) {
          // Load one tile of A and B into local memory.
          mm_Asub1[tileRow][tileColA] =
              mm_readA((globalRow - tileRow) / 2 + tileRow, globalColA);
          globalColA += TileInner;
          mm_Bsub1[tileRowB][tileCol] = mm_readB(globalRowB, globalCol);
          globalRowB += TileInner;
        }
      } else {
        if (tileRow < TileAOuter) {
          // Load one tile of A and B into local memory.
          mm_Asub1[tileRow][tileColA] =
              mm_readA((globalRow - tileRow) / 2 + tileRow, globalColA);
          globalColA += TileInner;
          mm_Bsub1[tileRowB][tileCol] = mm_readB(globalRowB, globalCol);
          globalRowB += TileInner;
        } else {
          // Compute acc values for a single thread.
          for (int k = 0; k < TileInner; k++) {
            acc += mm_Asub2[tileRow - TileAOuter][k] * mm_Bsub2[k][tileCol];
          }
        }
      }
      barrier();
      if (t != 0) {
        t++;
      }

      if (t < numTiles) {
        if (tileRow < TileAOuter) {
          // Load one tile of A and B into local memory.
          mm_Asub2[tileRow][tileColA] =
              mm_readA((globalRow - tileRow) / 2 + tileRow, globalColA);
          globalColA += TileInner;
          mm_Bsub2[tileRowB][tileCol] = mm_readB(globalRowB, globalCol);
          globalRowB += TileInner;
        } else {
          // Compute acc values for a single thread.
          for (int k = 0; k < TileInner; k++) {
            acc += mm_Asub1[tileRow - TileAOuter][k] * mm_Bsub1[k][tileCol];
          }
        }
      }
      barrier();
    }
    if (tileRow >= TileAOuter) {
      mm_write((globalRow - tileRow) / 2 + tileRow - TileAOuter,
          globalCol, acc);
    }
  }
  `;
}

export function makeMatMulSmallOutputSizeSourceWgsl(
    workGroupSize: [number, number, number]): string {
  const tileAOuter = workGroupSize[1] / 2;
  const tileBOuter = workGroupSize[0];
  const tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
  return `
  var<workgroup> mm_Asub1 : array<array<f32, ${tileInner}>, ${tileAOuter}>;
  var<workgroup> mm_Bsub1 : array<array<f32, ${tileBOuter}>, ${tileInner}>;
  var<workgroup> mm_Asub2 : array<array<f32, ${tileInner}>, ${tileAOuter}>;
  var<workgroup> mm_Bsub2 : array<array<f32, ${tileBOuter}>, ${tileInner}>;

  // If the output size is small for matrix multiplication, avoid to use vec4
  // and handle some elements per thread to optimally utilize the ALU.
  // Introduces two shared memory buffers, some logical threads could handle
  // arithmetic operations and others handle IO operations between barrier api,
  // makes ALUs and load/store units work simultaneously, could improves
  // the performance.
  ${getMainHeaderStringWgsl()} {
    let tileRow = i32(localId.y);
    let tileCol = i32(localId.x);
    let globalRow = i32(globalId.y);
    let globalCol = i32(globalId.x);

    // uniforms.dimInner should be greater than 0.
    let numTiles = (uniforms.dimInner - 1) / ${tileInner} + 1;
    var acc = 0.0;

    var globalColA = tileCol;
    var globalRowB = tileRow;
    for (var t = 0; t < numTiles; t = t + 1) {
      if (t == 0) {
        if (tileRow < ${tileAOuter}) {
          // Load one tile of A and B into local memory.
          // globalRow is always greater than or equal tileRow.
          mm_Asub1[tileRow][tileCol] =
              mm_readA((globalRow - tileRow) / 2 + tileRow, globalColA, vec3<i32>(globalId));
          globalColA = globalColA + ${tileInner};
          mm_Bsub1[tileRow][tileCol] = mm_readB(globalRowB, globalCol, vec3<i32>(globalId));
          globalRowB = globalRowB + ${tileInner};
        }
      } else {
        if (tileRow < ${tileAOuter}) {
          // Load one tile of A and B into local memory.
          // globalRow is always greater than or equal tileRow.
          mm_Asub1[tileRow][tileCol] =
              mm_readA((globalRow - tileRow) / 2 + tileRow, globalColA, vec3<i32>(globalId));
          globalColA = globalColA + ${tileInner};
          mm_Bsub1[tileRow][tileCol] = mm_readB(globalRowB, globalCol, vec3<i32>(globalId));
          globalRowB = globalRowB + ${tileInner};
        } else {
          // Compute acc values for a single thread.
          for (var k = 0; k < ${tileInner}; k = k + 1) {
            let subRow = i32(tileRow - ${tileAOuter});
            if (subRow < 0) {
              continue;
            }
            acc = acc + mm_Asub2[i32(subRow)][k] * mm_Bsub2[k][tileCol];
          }
        }
      }
      workgroupBarrier();
      if (t != 0) {
        t = t + 1;
      }

      if (t < numTiles) {
        if (tileRow < ${tileAOuter}) {
          // Load one tile of A and B into local memory.
          // globalRow is always greater than or equal tileRow.
          mm_Asub2[tileRow][tileCol] =
              mm_readA((globalRow - tileRow) / 2 + tileRow, globalColA, vec3<i32>(globalId));
          globalColA = globalColA + ${tileInner};
          mm_Bsub2[tileRow][tileCol] = mm_readB(globalRowB, globalCol, vec3<i32>(globalId));
          globalRowB = globalRowB + ${tileInner};
        } else {
          // Compute acc values for a single thread.
          for (var k = 0; k < ${tileInner}; k = k + 1) {
            let subRow = tileRow - ${tileAOuter};
            if (subRow < 0) {
              continue;
            }
            acc = acc + mm_Asub1[subRow][k] * mm_Bsub1[k][tileCol];
          }
        }
      }
      workgroupBarrier();
    }
    let writeCol = (globalRow - tileRow) / 2 + tileRow - ${tileAOuter};
    if (tileRow >= ${tileAOuter} && writeCol >= 0) {
      mm_write(writeCol, globalCol, acc, vec3<i32>(globalId));
    }
  }
  `;
}

export class MatMulSmallOutputSizeProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  uniformsWgsl = `dimAOuter : i32; dimBOuter : i32; dimInner : i32;`;
  workGroupSize: [number, number, number] = [8, 16, 1];
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  useWgsl: boolean;

  constructor(
      aShape: [number, number, number], bShape: [number, number, number],
      outputShape: [number, number, number], bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    util.assert(
        aShape[1] <= 16 || bShape[2] <= 16,
        () => 'This program can be only used when A width is small.');
    this.outputShape = outputShape;

    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    this.dispatch = [
      Math.ceil(outputShape[2] / this.workGroupSize[0]),
      Math.ceil(outputShape[1] * 2 / this.workGroupSize[1]), outputShape[0]
    ];

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
    this.shaderKey = `matMulSmallOutputSize_${this.activation}`;
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    const sampleA =
        `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            A[batch * batchASize + row * dimInner + col] : 0`;

    const sampleB =
        `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B[batch * batchBSize + row * dimBOuter + col] : 0`;

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(this.activation);
      if (this.hasPreluActivationWeights) {
        activationSnippet = `float activation(float a, ivec3 outCoord) {
            float b = getPreluActivationWeightsAtOutCoords(outCoord);
            ${activationOp}
            }`;
      } else {
        activationSnippet = `float activation(float a, ivec3 outCoord) {
            ${activationOp}
        }`;
      }

      applyActivationSnippet = 'value = activation(value, outCoord);';
    }

    const addBiasSnippet =
        this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';

    const userCode = `
      ${activationSnippet}

      int dimAOuter = aShape[1];
      int dimInner = aShape[2];
      int dimBOuter = bShape[2];
      int batch;
      ${makeMatMulSmallOutputSizeSource()}
      float mm_readA(int row, int col) {
        int batchASize = aShape[1] * aShape[2];
        return ${sampleA};
      }
      float mm_readB(int row, int col) {
        int batchBSize = bShape[1] * bShape[2];
        return ${sampleB};
      }
      void mm_write(int row, int col, float value) {
        if (coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimBOuter))) {
          ivec3 outCoord = ivec3(batch, row, col);
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutput(batch, row, col, value);
        }
      }
      void main() {
        batch = int(gl_GlobalInvocationID.z);
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    `;
    return userCode;
  }

  getUserCodeWgsl(): string {
    const sampleA =
        `if (coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimAOuter, uniforms.dimInner))) {
          return A.numbers[batch * batchASize + row * uniforms.dimInner + col]; 
        } 
        return 0.0;`;

    const sampleB =
        `if (coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
           return B.numbers[batch * batchBSize + row * uniforms.dimBOuter + col]; 
         }
         return 0.0;`;

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp =
          mapActivationToShaderProgram(this.activation, false, true);
      if (this.hasPreluActivationWeights) {
        activationSnippet =
            `fn activation(a : f32, outCoord : vec3<i32>) -> f32 {
            let b = getPreluActivationWeightsAtOutCoordsByCoords(outCoord);
            ${activationOp}
            }`;
      } else {
        activationSnippet =
            `fn activation(a : f32, outCoord : vec3<i32>) -> f32 {
            ${activationOp}
        }`;
      }

      applyActivationSnippet = 'value = activation(value, outCoord);';
    }

    const addBiasSnippet = this.addBias ?
        'value = value + getBiasAtOutCoordsByCoords(outCoord);' :
        '';

    const userCode = `
      ${activationSnippet}
      
      fn mm_readA(row : i32, col : i32,  globalId : vec3<i32>) -> f32 {
        let batchASize = uniforms.aShape[1] * uniforms.aShape[2];
        let batch = globalId.z;
        ${sampleA}
      }
      fn mm_readB(row : i32, col : i32,  globalId : vec3<i32>) -> f32 {
        let batch = globalId.z;
        let batchBSize = uniforms.bShape[1] * uniforms.bShape[2];
        ${sampleB}
      }
      fn mm_write(row : i32, col : i32, valueIn : f32, globalId : vec3<i32>) {
        if (coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimAOuter, uniforms.dimBOuter))) {
          let batch = globalId.z;
          let outCoord = vec3<i32>(batch, row, col);
          var value = valueIn;
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutput(batch, row, col, value);
        }
      }
      ${makeMatMulSmallOutputSizeSourceWgsl(this.workGroupSize)}
    `;
    return userCode;
  }
}
