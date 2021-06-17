/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {backend_util, TensorInfo} from '@tensorflow/tfjs-core';
import {computeDispatch, computeWorkGroupSizeForMatMul, tilesFitEvenlyIntoShape} from '../webgpu_util';
import {mapActivationToShaderProgram} from './activation_util';

import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export function makeMatMulPackedVec4Source(workPerThread: number[]): string {
  return `
    vec4 mm_readA(int row, int col);
    vec4 mm_readB(int row, int col);
    void mm_write(int row, int col, vec4 value);

    const int RowPerThread = ${workPerThread[1]};
    const int ColPerThread = ${
      workPerThread[0]}; // only support ColPerThread = 4
    const int TileAOuter = int(gl_WorkGroupSize.y) * RowPerThread;
    const int TileBOuter = int(gl_WorkGroupSize.x) * ColPerThread;
    const int TileInner = TileBOuter;

    shared vec4 mm_Asub[TileAOuter][TileInner / ColPerThread];
    shared vec4 mm_Bsub[TileInner][TileBOuter / ColPerThread];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
      int tileCol = int(gl_LocalInvocationID.x);

      int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
      int globalCol = int(gl_GlobalInvocationID.x);

      int numTiles = (dimInner - 1) / TileInner + 1;

      vec4 acc[RowPerThread];
      vec4 ACached;
      vec4 BCached[4];

      // Without this initialization strange values show up in acc.
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          acc[innerRow] = vec4(0.0, 0.0, 0.0, 0.0);
      }

      // Loop over shared dimension.
      int globalColA = tileCol;
      const int RowPerThreadB = TileInner / int(gl_WorkGroupSize.y);
      int tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
            int inputRow = tileRow + innerRow;
            int inputCol = tileCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                globalColA);
        }
        globalColA += TileInner / ColPerThread;

        // Load one tile of B into local memory.
        for (int innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
            int inputRow = tileRowB + innerRow;
            int inputCol = tileCol;

            mm_Bsub[inputRow][inputCol] = mm_readB(
              t * TileInner + inputRow,
              globalCol);
        }

        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < TileInner / ColPerThread; k++) {
          BCached[0] = mm_Bsub[k * ColPerThread][tileCol];
          BCached[1] = mm_Bsub[k * ColPerThread + 1][tileCol];
          BCached[2] = mm_Bsub[k * ColPerThread + 2][tileCol];
          BCached[3] = mm_Bsub[k * ColPerThread + 3][tileCol];

          for (int i = 0; i < RowPerThread; i++) {
            ACached = mm_Asub[tileRow + i][k];
            acc[i] = BCached[0] * ACached.x + acc[i];
            acc[i] = BCached[1] * ACached.y + acc[i];
            acc[i] = BCached[2] * ACached.z + acc[i];
            acc[i] = BCached[3] * ACached.w + acc[i];
          }
        }
        barrier();
      }

      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        mm_write(globalRow + innerRow,
          globalCol,
          acc[innerRow]);
      }
    }
  `;
}

function getSharedArray1DCodeWgsl() {
  return `
  var<workgroup> mm_Asub : array<vec4<f32>, 1024>;
  var<workgroup> mm_Bsub : array<vec4<f32>, 1024>;`;
}

function getSharedArray2DCodeWgsl() {
  return `
  var<workgroup> mm_Asub : array<array<vec4<f32>, 16>, 64>;
  var<workgroup> mm_Bsub : array<array<vec4<f32>, 16>, 64>;`;
}

function getA1DCodeWgsl() {
  return `
  let index : u32 = inputRow * TileInner / ColPerThread + inputCol;
  mm_Asub[index] = mm_readA(globalRow + innerRow, globalColA, global_id);`;
}

function getA2DCodeWgsl() {
  return `
  mm_Asub[inputRow][inputCol] = mm_readA(globalRow + innerRow, globalColA, global_id);`;
}

function getB1DCodeWgsl() {
  return `
  let index : u32 = inputRow * TileBOuter / ColPerThread + inputCol;
  mm_Bsub[index] = mm_readB(t * TileInner + inputRow, globalCol);`;
}

function getB2DCodeWgsl() {
  return `
  mm_Bsub[inputRow][inputCol] = mm_readB(t * TileInner + inputRow, globalCol);`;
}

function getCompute1DCodeWgsl() {
  return `
  // Compute acc values for a single thread.
  for (var k : u32 = 0u; k < TileInner / ColPerThread; k = k + 1u) {
      BCached[0] = mm_Bsub[(k * ColPerThread) * (TileBOuter / ColPerThread) + tileCol];
      BCached[1] = mm_Bsub[(k * ColPerThread + 1u) * (TileBOuter / ColPerThread) + tileCol];
      BCached[2] = mm_Bsub[(k * ColPerThread + 2u) * (TileBOuter / ColPerThread) + tileCol];
      BCached[3] = mm_Bsub[(k * ColPerThread + 3u) * (TileBOuter / ColPerThread) + tileCol];

      for (var i : u32 = 0u; i < RowPerThread; i = i + 1u) {
          ACached = mm_Asub[(tileRow + i) * (TileInner / ColPerThread) + k];

          acc[i] = BCached[0] * ACached.x + acc[i];
          acc[i] = BCached[1] * ACached.y + acc[i];
          acc[i] = BCached[2] * ACached.z + acc[i];
          acc[i] = BCached[3] * ACached.w + acc[i];
      }
  }`;
}

function getCompute2DCodeWgsl() {
  return `
  // Compute acc values for a single thread.
  for (var k : u32 = 0u; k < TileInner / ColPerThread; k = k + 1u) {
      BCached[0] = mm_Bsub[k * ColPerThread][tileCol];
      BCached[1] = mm_Bsub[k * ColPerThread + 1u][tileCol];
      BCached[2] = mm_Bsub[k * ColPerThread + 2u][tileCol];
      BCached[3] = mm_Bsub[k * ColPerThread + 3u][tileCol];

      for (var i : u32 = 0u; i < RowPerThread; i = i + 1u) {
          ACached = mm_Asub[tileRow + i][k];
          acc[i] = BCached[0] * ACached.x + acc[i];
          acc[i] = BCached[1] * ACached.y + acc[i];
          acc[i] = BCached[2] * ACached.z + acc[i];
          acc[i] = BCached[3] * ACached.w + acc[i];
      }
  }`;
}

function getMainCodeWgsl(getA: string, getB: string, computeAcc: string) {
  return `
  let RowPerThread : u32 = 4u;
  let ColPerThread : u32 = 4u;
  let TileAOuter : u32 = 64u;
  let TileBOuter : u32 = 64u;
  let TileInner : u32 = 64u;

  [[stage(compute), workgroup_size(16, 16, 1)]]
  fn main([[builtin(local_invocation_id)]] local_id : vec3<u32>,
        [[builtin(global_invocation_id)]] global_id  : vec3<u32>) {

    let tileRow : u32 = local_id.y * RowPerThread;
    let tileCol : u32 = local_id.x;

    let globalRow : u32 = global_id.y * RowPerThread;
    let globalCol : u32 = global_id.x;
    var dimInner : u32 = uniforms.filterDims[0] * uniforms.filterDims[1] * uniforms.xShape[3];
    let numTiles : u32 = (dimInner - 1u) / TileInner + 1u;

    var acc: array<vec4<f32>, 4>;
    var ACached : vec4<f32>;
    var BCached : array<vec4<f32>, 4>;

    // Without this initialization strange values show up in acc.
    // TODO: Remove it once the following bug is fixed.
    // https://bugs.chromium.org/p/tint/issues/detail?id=759
    for (var index : u32 = 0u; index < RowPerThread; index = index + 1u) {
        acc[index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    var globalColA : u32 = tileCol;
    let RowPerThreadB : u32 = TileInner / 16u;
    let tileRowB : u32 = local_id.y * RowPerThreadB;

    // Loop over shared dimension.
    for (var t : u32 = 0u; t < numTiles; t = t + 1u) {
        // Load one tile of A into local memory.
        for (var innerRow : u32 = 0u; innerRow < RowPerThread; innerRow = innerRow + 1u) {
            let inputRow : u32 = tileRow + innerRow;
            let inputCol : u32 = tileCol;
            ${getA}
        }
        globalColA = globalColA + TileInner / ColPerThread;

        // Load one tile of B into local memory.
        for (var innerRow : u32 = 0u; innerRow < RowPerThreadB; innerRow = innerRow + 1u) {
            let inputRow : u32 = tileRowB + innerRow;
            let inputCol : u32 = tileCol;
            ${getB}
        }

        workgroupBarrier();

        ${computeAcc}

        workgroupBarrier();
    }

    for (var innerRow : u32 = 0u; innerRow < RowPerThread; innerRow = innerRow + 1u) {
        mm_write(globalRow + innerRow,
                 globalCol,
                 acc[innerRow], global_id);
    }
}`;
}

export function makeMatMulPackedVec4SourceWgsl(
    addBiasSnippet: string, applyActivationSnippet: string,
    workPerThread: number[]): string {
  const kMatMulVec4TwoDimensionalSharedArray = getSharedArray2DCodeWgsl() +
      getMainCodeWgsl(getA2DCodeWgsl(), getB2DCodeWgsl(),
                      getCompute2DCodeWgsl());
  return kMatMulVec4TwoDimensionalSharedArray;
}

export function makeMatMulVectorVec4SourceWgsl(
    addBiasSnippet: string, applyActivationSnippet: string): string {
  const kMatMulVec4OneDimensionalSharedArray = getSharedArray1DCodeWgsl() +
      getMainCodeWgsl(getA1DCodeWgsl(), getB1DCodeWgsl(),
                      getCompute1DCodeWgsl());

  return kMatMulVec4OneDimensionalSharedArray;
}

export function makeMatMulVectorVec4Source(): string {
  return `
    vec4 mm_readA(int row, int col);
    vec4 mm_readB(int row, int col);
    void mm_write(int row, int col, vec4 value);

    const int TileSize = int(gl_WorkGroupSize.x) * 4;

    shared vec4 mm_Asub[TileSize / 4];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileCol = int(gl_LocalInvocationID.x);
      int globalCol = int(gl_GlobalInvocationID.x);
      int globalRow = int(gl_GlobalInvocationID.y);

      int numTiles = (dimInner - 1) / TileSize + 1;

      // Without this initialization strange values show up in acc.
      vec4 acc = vec4(0.0);

      // Loop over shared dimension.
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        int colA = t * TileSize / 4 + tileCol;
        mm_Asub[tileCol] = mm_readA(globalRow, colA);
        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < TileSize / 4; k++) {
          int rowB = t * TileSize + k * 4;
          vec4 BCached0 = mm_readB(rowB, globalCol);
          vec4 BCached1 = mm_readB(rowB + 1, globalCol);
          vec4 BCached2 = mm_readB(rowB + 2, globalCol);
          vec4 BCached3 = mm_readB(rowB + 3, globalCol);

          vec4 ACached = mm_Asub[k];
          acc += BCached0 * ACached.x;
          acc += BCached1 * ACached.y;
          acc += BCached2 * ACached.z;
          acc += BCached3 * ACached.w;
        }

        barrier();
      }

      if (globalRow < dimAOuter && globalCol < dimBOuter) {
        mm_write(globalRow, globalCol, acc);
      }
    }
  `;
}

export class MatMulPackedVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  workGroupSize: [number, number, number] = [16, 16, 1];
  useWgsl: boolean;
  isVec4 = true;
  aShape: [number, number, number];
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  vecSize = 4;
  fitA: boolean;
  fitB: boolean;

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      rowPerThread: number, bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;
    this.workGroupSize = computeWorkGroupSizeForMatMul(
        outputShape[1], aShape[2], outputShape[2]);
    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    if (outputShape[1] === 1) {
      rowPerThread = 1;
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [this.vecSize, rowPerThread, 1]);

    const addBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.workPerThread = rowPerThread;
    this.aShape = aShape;
    this.addBias = addBias;
    this.useWgsl = getUseWgsl();
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;

    [this.fitA, this.fitB] = this.getShapeFit();

    this.shaderKey = `matMulPackedVec4_${rowPerThread}_${this.activation}_${
        this.fitA}_${this.fitB}_${this.outputShape[1] > 1}`;
  }

  getShapeFit(): boolean[] {
    const dimInner = this.aShape[2];
    const dimBOuter = this.outputShape[2];
    const bShape = [this.outputShape[0], dimInner, dimBOuter];
    const tileAOuter = this.workGroupSize[1] * this.workPerThread;
    const tileBOuter = this.workGroupSize[0] * this.vecSize;
    const tileInner = tileBOuter;  // Make sure tileInner is divisible by 4.

    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];
    return [
      tilesFitEvenlyIntoShape(tileSizeA, this.aShape.slice(1)),
      tilesFitEvenlyIntoShape(tileSizeB, bShape.slice(1))
    ];
  }

  getUserCode(): string {
    const sampleA = this.fitA ?
        `A[batch * batchASize + row * dimInner / 4 + col]` :
        `coordsInBounds(ivec2(row, col * 4), ivec2(dimAOuter, dimInner)) ?
            A[batch * batchASize + row * dimInner / 4 + col] :
            vec4(0.0, 0.0, 0.0, 0.0)`;

    const sampleB = this.fitB ?
        `B[batch * batchBSize + row * dimBOuter / 4 + col]` :
        `coordsInBounds(ivec2(row, col * 4), ivec2(dimInner, dimBOuter)) ?
            B[batch * batchBSize + row * dimBOuter / 4 + col] :
            vec4(0.0, 0.0, 0.0, 0.0)`;

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(
          this.activation, this.isVec4, this.useWgsl);
      if (this.hasPreluActivationWeights) {
        activationSnippet = `vec4 activation(vec4 a, ivec3 outCoord) {
                  vec4 b = getPreluActivationWeightsAtOutCoords(outCoord);
                  ${activationOp}
                }`;
      } else {
        activationSnippet = `
                vec4 activation(vec4 a, ivec3 outCoord) {
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

      ${
        this.outputShape[1] > 1 ?
            makeMatMulPackedVec4Source([this.vecSize, this.workPerThread, 1]) :
            makeMatMulVectorVec4Source()}

      vec4 mm_readA(int row, int col) {
        int batchASize = aShape[1] * aShape[2] / ${this.vecSize};
        return ${sampleA};
      }

      vec4 mm_readB(int row, int col) {
        // TODO: This is not covered in unit tests.
        int batchBSize = bShape[1] * bShape[2] / ${this.vecSize};
        return ${sampleB};
      }

      void mm_write(int row, int col, vec4 value) {
        if (row < dimAOuter && col * 4 < dimBOuter)
        {
          ivec3 outCoord = ivec3(batch, row, col * 4);
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutput(outCoord[0], outCoord[1], outCoord[2], value);
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
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      if (this.hasPreluActivationWeights) {
        activationSnippet =
            `fn activation(a : vec4<f32>, outCoord :  vec3<i32>) -> vec4<f32>{
                  vec4 b = getPreluActivationWeightsAtOutCoords2(outCoord);
                  ${this.activation}
                }`;
      } else {
        activationSnippet = `
                fn activation(a : vec4<f32>, outCoord :  vec3<i32>) -> vec4<f32> {
                  ${this.activation}
                }`;
      }

      applyActivationSnippet = 'value = activation(value, outCoord);';
    }

    const addBiasSnippet =
        this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';

    const userCode = `
      ${activationSnippet}

      ${
        this.outputShape[1] > 1 ? makeMatMulPackedVec4SourceWgsl(
                                      addBiasSnippet, applyActivationSnippet,
                                      [this.vecSize, this.workPerThread, 1]) :
                                  makeMatMulVectorVec4SourceWgsl(
                                      addBiasSnippet, applyActivationSnippet)}
    `;
    return userCode;
  }
}
