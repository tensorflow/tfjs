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

import {getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
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
          acc[innerRow] = vec4(0.0);
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

export function makeMatMulPackedVec4SourceWgsl(
    workPerThread: number[], workGroupSize: [number, number, number]): string {
  const tileInfo = {
    RowPerThread: workPerThread[1],
    ColPerThread: workPerThread[0],
    TileAOuter: workGroupSize[1] * workPerThread[1],
    TileBOuter: workGroupSize[0] * workPerThread[0],
    TileInner: workGroupSize[0] * workPerThread[0]
  };
  return `
  var<workgroup> mm_Asub : array<array<vec4<f32>, ${
      tileInfo.TileInner / tileInfo.ColPerThread}>, ${tileInfo.TileAOuter}>;
  var<workgroup> mm_Bsub : array<array<vec4<f32>, ${
      tileInfo.TileBOuter / tileInfo.ColPerThread}>, ${tileInfo.TileInner}>;

  let RowPerThread = ${tileInfo.RowPerThread}u;
  let ColPerThread = ${tileInfo.ColPerThread}u; // only support ColPerThread = 4
  let TileAOuter = ${tileInfo.TileAOuter}u;
  let TileBOuter = ${tileInfo.TileBOuter}u;
  let TileInner = ${tileInfo.TileInner}u;

  ${getWorkGroupSizeStringWgsl(workGroupSize)}
  fn main([[builtin(local_invocation_id)]] localId : vec3<u32>,
        [[builtin(global_invocation_id)]] globalId : vec3<u32>) {

    let tileRow = localId.y * RowPerThread;
    let tileCol = localId.x;

    let globalRow = globalId.y * RowPerThread;
    let globalCol = globalId.x;
    let numTiles = (uniforms.dimInner - 1u) / TileInner + 1u;

    var acc: array<vec4<f32>, ${tileInfo.RowPerThread}>;
    var ACached : vec4<f32>;
    var BCached : array<vec4<f32>, 4>;

    // Loop over shared dimension.
    var globalColA = tileCol;
    let RowPerThreadB = TileInner / ${workGroupSize[1]}u;
    let tileRowB = localId.y * RowPerThreadB;
    for (var t = 0u; t < numTiles; t = t + 1u) {
        // Load one tile of A into local memory.
        for (var innerRow = 0u; innerRow < RowPerThread; innerRow = innerRow + 1u) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileCol;
            mm_Asub[inputRow][inputCol] = mm_readA(globalRow + innerRow, globalColA, globalId);
        }
        globalColA = globalColA + TileInner / ColPerThread;

        // Load one tile of B into local memory.
        for (var innerRow = 0u; innerRow < RowPerThreadB; innerRow = innerRow + 1u) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(t * TileInner + inputRow, globalCol, globalId);
        }

        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0u; k < TileInner / ColPerThread; k = k + 1u) {
            BCached[0] = mm_Bsub[k * ColPerThread][tileCol];
            BCached[1] = mm_Bsub[k * ColPerThread + 1u][tileCol];
            BCached[2] = mm_Bsub[k * ColPerThread + 2u][tileCol];
            BCached[3] = mm_Bsub[k * ColPerThread + 3u][tileCol];

            for (var i = 0u; i < RowPerThread; i = i + 1u) {
                ACached = mm_Asub[tileRow + i][k];
                acc[i] = BCached[0] * ACached.x + acc[i];
                acc[i] = BCached[1] * ACached.y + acc[i];
                acc[i] = BCached[2] * ACached.z + acc[i];
                acc[i] = BCached[3] * ACached.w + acc[i];
            }
        }

        workgroupBarrier();
    }

    for (var innerRow = 0u; innerRow < RowPerThread; innerRow = innerRow + 1u) {
        mm_write(globalRow + innerRow,
                 globalCol,
                 acc[innerRow], globalId);
    }
}`;
}

export function makeMatMulVectorVec4SourceWgsl(
    workGroupSize: [number, number, number]): string {
  return `
  var<workgroup> mm_Asub : array<vec4<f32>, ${workGroupSize[0]}>;
  let tileSize = ${workGroupSize[0] * 4}u;
  ${getWorkGroupSizeStringWgsl(workGroupSize)}
  fn main([[builtin(local_invocation_id)]] localId : vec3<u32>,
        [[builtin(global_invocation_id)]] globalId : vec3<u32>) {
    let tileCol = localId.x;
    let globalCol = globalId.x;
    let globalRow = globalId.y;

    let numTiles = (uniforms.dimInner - 1u) / tileSize + 1u;

    // Without this initialization strange values show up in acc.
    var acc = vec4<f32>(0.0);

    // Loop over shared dimension.
    for (var t = 0u; t < numTiles; t = t + 1u) {
      // Load one tile of A into local memory.
      let colA = t * tileSize / 4u + tileCol;
      mm_Asub[tileCol] = mm_readA(globalRow, colA, globalId);

      workgroupBarrier();

      // Compute acc values for a single thread.
      for (var k = 0u; k < tileSize / 4u; k = k + 1u) {
        let rowB = t * tileSize + k * 4u;
        let BCached0 = mm_readB(rowB, globalCol, globalId);
        let BCached1 = mm_readB(rowB + 1u, globalCol, globalId);
        let BCached2 = mm_readB(rowB + 2u, globalCol, globalId);
        let BCached3 = mm_readB(rowB + 3u, globalCol, globalId);

        let ACached = mm_Asub[k];
        acc = acc + BCached0 * ACached.x;
        acc = acc + BCached1 * ACached.y;
        acc = acc + BCached2 * ACached.z;
        acc = acc + BCached3 * ACached.w;
      }

      workgroupBarrier();
    }

    if (globalRow < uniforms.dimAOuter && globalCol < uniforms.dimBOuter) {
      mm_write(globalRow, globalCol, acc, globalId);
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
  uniformsWgsl = `dimAOuter : u32; dimBOuter : u32; dimInner : u32;`;
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
            vec4(0.0)`;

    const sampleB = this.fitB ?
        `B[batch * batchBSize + row * dimBOuter / 4 + col]` :
        `coordsInBounds(ivec2(row, col * 4), ivec2(dimInner, dimBOuter)) ?
            B[batch * batchBSize + row * dimBOuter / 4 + col] :
            vec4(0.0)`;

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
    const sampleA = this.fitA ?
        `return A.numbers[batch * batchASize + row * uniforms.dimInner / 4u + col]` :
        `if (coordsInBounds2D(vec2<u32>(row, col * 4u), vec2<u32>(uniforms.dimAOuter, uniforms.dimInner))) {
            return A.numbers[batch * batchASize + row * uniforms.dimInner / 4u + col];
        }
        return vec4<f32>(0.0)`;

    const sampleB = this.fitB ?
        `return B.numbers[batch * batchBSize + row * uniforms.dimBOuter / 4u + col]` :
        `if(coordsInBounds2D(vec2<u32>(row, col * 4u), vec2<u32>(uniforms.dimInner, uniforms.dimBOuter))) {
             return B.numbers[batch * batchBSize + row * uniforms.dimBOuter / 4u + col];
        }
        return vec4<f32>(0.0)`;

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(
          this.activation, this.isVec4, this.useWgsl);
      if (this.hasPreluActivationWeights) {
        activationSnippet =
            `fn activation(a : vec4<f32>, outCoord : vec3<u32>) -> vec4<f32> {
                  let b = getPreluActivationWeightsAtOutCoordsByCoords(outCoord);
                  ${activationOp}
                }`;
      } else {
        activationSnippet = `
            fn activation(a : vec4<f32>, outCoord : vec3<u32>) -> vec4<f32> {
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
      fn mm_readA(row : u32, col : u32,  globalId : vec3<u32>) -> vec4<f32> {
        let batchASize = uniforms.aShape[1] * uniforms.aShape[2] / ${
        this.vecSize}u;
        let batch = globalId.z;
        ${sampleA};
      }

      fn mm_readB(row : u32, col : u32,  globalId : vec3<u32>) -> vec4<f32> {
        let batchBSize = uniforms.bShape[1] * uniforms.bShape[2] / ${
        this.vecSize}u;
        let batch = globalId.z;
        ${sampleB};
      }

      fn mm_write(row : u32, col : u32, valueIn : vec4<f32>, globalId : vec3<u32>) {
        if (row < uniforms.aShape[1] && col * 4u < uniforms.bShape[2])
        {
          var value = valueIn;
          let batch = globalId.z;
          let outCoord = vec3<u32>(batch, row, col * 4u);
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutput(outCoord[0], outCoord[1], outCoord[2], value);
        }
      }
      ${
        this.outputShape[1] > 1 ?
            makeMatMulPackedVec4SourceWgsl(
                [this.vecSize, this.workPerThread, 1], this.workGroupSize) :
            makeMatMulVectorVec4SourceWgsl(this.workGroupSize)}

    `;

    return userCode;
  }
}
