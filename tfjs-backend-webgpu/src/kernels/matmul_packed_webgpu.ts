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

import {backend_util, TensorInfo, util} from '@tensorflow/tfjs-core';

import {getMainHeaderStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, computeWorkGroupSizeForMatMul, tilesFitEvenlyIntoShape} from '../webgpu_util';

import {mapActivationToShaderProgram} from './activation_util';
import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export function makeMatMulPackedSource(workPerThread: number[]): string {
  return `
    float mm_readA(int row, int col);
    float mm_readB(int row, int col);
    void mm_write(int row, int col, float value);
    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter);

    const int RowPerThread = ${workPerThread[1]};
    const int ColPerThread = ${workPerThread[0]};
    const int TileAOuter = int(gl_WorkGroupSize.y) * RowPerThread;
    const int TileBOuter = int(gl_WorkGroupSize.x) * ColPerThread;
    const int TileInner = TileAOuter > TileBOuter ? TileAOuter : TileBOuter;

    shared float mm_Asub[TileAOuter][TileInner];
    shared float mm_Bsub[TileInner][TileBOuter];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileRow = int(gl_LocalInvocationID.y) * RowPerThread;
      int tileCol = int(gl_LocalInvocationID.x) * ColPerThread;

      int globalRow = int(gl_GlobalInvocationID.y) * RowPerThread;
      int globalCol = int(gl_GlobalInvocationID.x) * ColPerThread;

      int numTiles = (dimInner - 1) / TileInner + 1;

      float acc[RowPerThread][ColPerThread];
      float ACached;
      float BCached[ColPerThread];

      // Without this initialization strange values show up in acc.
      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
          acc[innerRow][innerCol] = 0.0;
        }
      }

      const int ColPerThreadA = TileInner / int(gl_WorkGroupSize.x);
      int tileColA = int(gl_LocalInvocationID.x) * ColPerThreadA;
      const int RowPerThreadB = TileInner / int(gl_WorkGroupSize.y);
      int tileRowB = int(gl_LocalInvocationID.y) * RowPerThreadB;

      // Loop over shared dimension.
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
          for (int innerCol = 0; innerCol < ColPerThreadA; innerCol++) {
            int inputRow = tileRow + innerRow;
            int inputCol = tileColA + innerCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                t * TileInner + inputCol);
          }
        }
        // Load one tile of B into local memory.
        for (int innerRow = 0; innerRow < RowPerThreadB; innerRow++) {
          for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
            int inputRow = tileRowB + innerRow;
            int inputCol = tileCol + innerCol;

            mm_Bsub[inputRow][inputCol] = mm_readB(
              t * TileInner + inputRow,
              globalCol + innerCol);;
          }
        }

        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < TileInner; k++) {
          for (int inner = 0; inner < ColPerThread; inner++) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
          }

          for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
            ACached = mm_Asub[tileRow + innerRow][k];
            for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {
              acc[innerRow][innerCol] += ACached * BCached[innerCol];
            }
          }
        }

        barrier();
      }

      for (int innerRow = 0; innerRow < RowPerThread; innerRow++) {
        for (int innerCol = 0; innerCol < ColPerThread; innerCol++) {

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

export function makeMatMulVectorSource(): string {
  return `
    float mm_readA(int row, int col);
    float mm_readB(int row, int col);
    void mm_write(int row, int col, float value);
    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter);

    const int TileSize = int(gl_WorkGroupSize.x) * 4;

    shared vec4 mm_Asub[TileSize / 4];

    void mm_matMul(int dimAOuter, int dimInner, int dimBOuter) {
      int tileCol = int(gl_LocalInvocationID.x);
      int globalCol = int(gl_GlobalInvocationID.x);
      int globalRow = int(gl_GlobalInvocationID.y);

      int numTiles = (dimInner - 1) / TileSize + 1;

      // Without this initialization strange values show up in acc.
      float acc = 0.0;

      // Loop over shared dimension.
      for (int t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        int colA = t * TileSize + tileCol * 4;
        mm_Asub[tileCol] = vec4(mm_readA(globalRow, colA),
                                mm_readA(globalRow, colA + 1),
                                mm_readA(globalRow, colA + 2),
                                mm_readA(globalRow, colA + 3));
        barrier();

        // Compute acc values for a single thread.
        for (int k = 0; k < TileSize / 4; k++) {
          int rowB = t * TileSize + k * 4;
          vec4 BCached = vec4(mm_readB(rowB, globalCol),
                              mm_readB(rowB + 1, globalCol),
                              mm_readB(rowB + 2, globalCol),
                              mm_readB(rowB + 3, globalCol));

          vec4 ACached = mm_Asub[k];
          acc += dot(ACached, BCached);
        }

        barrier();
      }

      if (globalRow < dimAOuter && globalCol < dimBOuter) {
        mm_write(globalRow, globalCol, acc);
      }
    }
  `;
}

export function makeMatMulPackedSourceWgsl(
    workPerThread: number[], workGroupSize: [number, number, number]): string {
  const tileAOuter = workGroupSize[1] * workPerThread[1];
  const tileBOuter = workGroupSize[0] * workPerThread[0];
  const tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
  return `
    var<workgroup> mm_Asub : array<array<f32, ${tileInner}>, ${tileAOuter}>;
    var<workgroup> mm_Bsub : array<array<f32, ${tileBOuter}>, ${tileInner}>;
    ${getMainHeaderStringWgsl()} {
      let tileRow = i32(localId.y) * ${workPerThread[1]};
      let tileCol = i32(localId.x) * ${workPerThread[0]};

      let globalRow = i32(globalId.y) * ${workPerThread[1]};
      let globalCol = i32(globalId.x) * ${workPerThread[0]};

      let numTiles = (uniforms.dimInner - 1) / ${tileInner} + 1;

      var acc : array<array<f32, ${workPerThread[0]}>, ${workPerThread[1]}>;
      var ACached : f32;
      var BCached : array<f32, ${workPerThread[0]}>;

      // Without this initialization strange values show up in acc.
      for (var innerRow = 0; innerRow < ${
      workPerThread[1]}; innerRow = innerRow + 1) {
        for (var innerCol = 0; innerCol < ${
      workPerThread[0]}; innerCol = innerCol + 1) {
          acc[innerRow][innerCol] = 0.0;
        }
      }

      let ColPerThreadA = ${tileInner} / ${workGroupSize[0]};
      let tileColA = i32(localId.x) * ColPerThreadA;
      let RowPerThreadB = ${tileInner} / ${workGroupSize[1]};
      let tileRowB = i32(localId.y) * RowPerThreadB;

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t = t + 1) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < ${
      workPerThread[1]}; innerRow = innerRow + 1) {
          for (var innerCol = 0; innerCol < ColPerThreadA; innerCol = innerCol + 1) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileColA + innerCol;

            mm_Asub[inputRow][inputCol] = mm_readA(
                globalRow + innerRow,
                t * ${tileInner} + inputCol, globalId);
          }
        }
        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < RowPerThreadB; innerRow = innerRow + 1) {
          for (var innerCol = 0; innerCol < ${
      workPerThread[0]}; innerCol = innerCol + 1) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol + innerCol;

            mm_Bsub[inputRow][inputCol] = mm_readB(
              t * ${tileInner} + inputRow,
              globalCol + innerCol, globalId);
          }
        }

        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < ${tileInner}; k = k + 1) {
          for (var inner = 0; inner < ${workPerThread[0]}; inner = inner + 1) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
          }

          for (var innerRow = 0; innerRow < ${
      workPerThread[1]}; innerRow = innerRow + 1) {
            ACached = mm_Asub[tileRow + innerRow][k];
            for (var innerCol = 0; innerCol < ${
      workPerThread[0]}; innerCol = innerCol + 1) {
              acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];
            }
          }
        }

        workgroupBarrier();
      }

      for (var innerRow = 0; innerRow < ${
      workPerThread[1]}; innerRow = innerRow + 1) {
        for (var innerCol = 0; innerCol < ${
      workPerThread[0]}; innerCol = innerCol + 1) {

          if ((globalCol + innerCol) < uniforms.dimBOuter &&
              (globalRow + innerRow) < uniforms.dimAOuter) {
            mm_write(globalRow + innerRow,
                     globalCol + innerCol,
                     acc[innerRow][innerCol], globalId);
          }
        }
      }
    }
  `;
}

export function makeMatMulVectorSourceWgsl(
    workGroupSize: [number, number, number]): string {
  return `
    let TileSize = ${workGroupSize[0] * 4};
    var<workgroup> mm_Asub : array<vec4<f32>, ${workGroupSize[0]}>;

    ${getMainHeaderStringWgsl()} {
      let tileCol = i32(localId.x);
      let globalCol = i32(globalId.x);
      let globalRow = i32(globalId.y);

      let numTiles = (uniforms.dimInner - 1) / TileSize + 1;

      // Without this initialization strange values show up in acc.
      var acc = 0.0;

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t = t + 1) {
        // Load one tile of A into local memory.
        let colA = t * TileSize + tileCol * 4;
        mm_Asub[tileCol] = vec4<f32>(mm_readA(globalRow, colA, globalId),
                                mm_readA(globalRow, colA + 1, globalId),
                                mm_readA(globalRow, colA + 2, globalId),
                                mm_readA(globalRow, colA + 3, globalId));
        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < TileSize / 4; k = k + 1) {
          let rowB = t * TileSize + k * 4;
          let BCached = vec4<f32>(mm_readB(rowB, globalCol, globalId),
                              mm_readB(rowB + 1, globalCol, globalId),
                              mm_readB(rowB + 2, globalCol, globalId),
                              mm_readB(rowB + 3, globalCol, globalId));

          let ACached = mm_Asub[k];
          acc = acc + dot(ACached, BCached);
        }

        workgroupBarrier();
      }

      if (globalRow < uniforms.dimAOuter && globalCol < uniforms.dimBOuter) {
        mm_write(globalRow, globalCol, acc, globalId);
      }
    }
  `;
}

export class MatMulPackedProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  workPerThread: number;
  variableNames = ['A', 'B'];
  uniformsWgsl = `dimAOuter : i32; dimBOuter : i32; dimInner : i32;`;
  workGroupSize: [number, number, number] = [16, 16, 1];
  useWgsl: boolean;
  aShape: [number, number, number];
  transposeA: boolean;
  transposeB: boolean;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  fitA: boolean;
  fitB: boolean;

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      workPerThread: number, transposeA = false, transposeB = false,
      bias: TensorInfo = null, activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0]};
    const dimInner = transposeA ? aShape[1] : aShape[2];
    this.workGroupSize =
        computeWorkGroupSizeForMatMul(outputShape[1], dimInner, outputShape[2]);
    if (outputShape[1] === 1 || outputShape[2] === 1) {
      workPerThread = 1;
    }
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        [workPerThread, workPerThread, 1]);
    // If dispaching number is one, it means only one work group is running.
    // For modern GPUs, it supports multiple work groups running in parallel.
    // So there may be some idle hardware threads.
    // In this case, we prefer to reduce the work per thread and improve the
    // thread utilization
    if (util.arraysEqual(this.dispatch, [1, 1, 1])) {
      workPerThread = 1;
      this.dispatch = computeDispatch(
          this.dispatchLayout, this.outputShape, this.workGroupSize,
          [workPerThread, workPerThread, 1]);
    }
    const addBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.workPerThread = workPerThread;
    this.aShape = aShape;
    this.transposeA = transposeA;
    this.transposeB = transposeB;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.useWgsl = getUseWgsl();

    const dimBOuter = this.outputShape[2];
    const bShape = this.transposeB ?
        [this.outputShape[0], dimBOuter, dimInner] :
        [this.outputShape[0], dimInner, dimBOuter];

    [this.fitA, this.fitB] = this.getShapeFit(bShape);
    this.shaderKey = `matMulPacked_${this.workPerThread}_${transposeA}_${
        transposeB}_${this.activation}_${this.fitA}_${this.fitB}_${
        this.outputShape[1] > 1}`;
  }

  getShapeFit(bShape: number[]): boolean[] {
    const tileAOuter = this.workGroupSize[1] * this.workPerThread;
    const tileBOuter = this.workGroupSize[0] * this.workPerThread;
    let tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
    if (this.outputShape[1] === 1) {
      tileInner *= 4;
    }
    util.assert(
        tileInner % this.workGroupSize[0] === 0 &&
            tileInner % this.workGroupSize[1] === 0,
        () => `tileInner must be multiple of workgroupsize.x ` +
            `and workgroupsize.y`);
    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];

    return [
      tilesFitEvenlyIntoShape(tileSizeA, this.aShape.slice(1)),
      tilesFitEvenlyIntoShape(tileSizeB, bShape.slice(1))
    ];
  }

  getUserCode(): string {
    let sampleA;

    if (this.transposeA === false) {
      sampleA = this.fitA ?
          `A[batch * batchASize + row * dimInner + col]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            A[batch * batchASize + row * dimInner + col] : 0`;
    } else {
      sampleA = this.fitA ?
          `A[batch * batchASize + col * dimAOuter + row]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimAOuter, dimInner)) ?
            A[batch* batchASize + col * dimAOuter + row] : 0`;
    }

    let sampleB;
    if (this.transposeB === false) {
      sampleB = this.fitB ?
          `B[batch * batchBSize + row * dimBOuter + col]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B[batch * batchBSize + row * dimBOuter + col] : 0`;
    } else {
      sampleB = this.fitB ?
          `B[batch * batchBSize + col * dimInner + row]` :
          `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
            B[batch * batchBSize + col * dimInner + row] : 0`;
    }

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp =
          mapActivationToShaderProgram(this.activation, false, this.useWgsl);
      if (this.hasPreluActivationWeights) {
        activationSnippet = `float activation(float a, ivec3 outCoord) {
              float b = getPreluActivationWeightsAtOutCoords(outCoord);
              ${activationOp}
            }`;
      } else {
        activationSnippet = `
              float activation(float a, ivec3 outCoord) {
                ${activationOp}
              }
            `;
      }

      applyActivationSnippet = 'value = activation(value, outCoord);';
    }

    const addBiasSnippet =
        this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';

    const userCode = `
      ${activationSnippet}

      int dimAOuter = ${this.transposeA === true ? `aShape[2]` : `aShape[1]`};
      int dimInner = ${this.transposeA === true ? `aShape[1]` : `aShape[2]`};
      int dimBOuter = ${this.transposeB === true ? `bShape[1]` : `bShape[2]`};

      int batch;

      ${
        this.outputShape[1] > 1 ?
            makeMatMulPackedSource(
                [this.workPerThread, this.workPerThread, 1]) :
            makeMatMulVectorSource()}
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
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(batch, row, col, value);
      }
      void main() {
        batch = int(gl_GlobalInvocationID.z);
        mm_matMul(dimAOuter, dimInner, dimBOuter);
      }
    `;
    return userCode;
  }

  getUserCodeWgsl(): string {
    let sampleA;

    if (this.transposeA === false) {
      sampleA = this.fitA ?
          `return A.numbers[batch * batchASize + row * uniforms.dimInner + col];` :
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimAOuter, uniforms.dimInner))) {
             return A.numbers[batch * batchASize + row * uniforms.dimInner + col];
           }
           return 0.0;`;
    } else {
      sampleA = this.fitA ?
          `return A.numbers[batch * batchASize + col * uniforms.dimAOuter + row];` :
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimAOuter, uniforms.dimInner))) {
             return A.numbers[batch* batchASize + col * uniforms.dimAOuter + row];
           }
           return 0.0;`;
    }

    let sampleB;
    if (this.transposeB === false) {
      sampleB = this.fitB ?
          `return B.numbers[batch * batchBSize + row * uniforms.dimBOuter + col];` :
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
             return B.numbers[batch * batchBSize + row * uniforms.dimBOuter + col];
           }
           return 0.0;`;
    } else {
      sampleB = this.fitB ?
          `return B.numbers[batch * batchBSize + col * uniforms.dimInner + row];` :
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
             return B.numbers[batch * batchBSize + col * uniforms.dimInner + row];
           }
           return 0.0;`;
    }

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp =
          mapActivationToShaderProgram(this.activation, false, this.useWgsl);
      if (this.hasPreluActivationWeights) {
        activationSnippet =
            `fn activation(a : f32, outCoord : vec3<i32>) -> f32 {
               let b = getPreluActivationWeightsAtOutCoordsByCoords(outCoord);
               ${activationOp}
            }`;
      } else {
        activationSnippet = `
              fn activation(a : f32, outCoord : vec3<i32>) -> f32 {
                ${activationOp}
              }
            `;
      }

      applyActivationSnippet = 'value = activation(value, outCoord);';
    }

    const addBiasSnippet = this.addBias ?
        'value = value + getBiasAtOutCoordsByCoords(outCoord);' :
        '';

    const userCode = `
      ${activationSnippet}

      fn mm_readA(row : i32, col : i32,  globalId : vec3<u32>) -> f32 {
        let batchASize = uniforms.aShape[1] * uniforms.aShape[2];
        let batch = i32(globalId.z);
        ${sampleA}
      }

      fn mm_readB(row : i32, col : i32,  globalId : vec3<u32>) -> f32 {
        let batch = i32(globalId.z);
        let batchBSize = uniforms.bShape[1] * uniforms.bShape[2];
        ${sampleB}
      }

      fn mm_write(row : i32, col : i32, valueIn : f32, globalId : vec3<u32>) {
        var value = valueIn;
        let batch = i32(globalId.z);
        let outCoord = vec3<i32>(batch, row, col);
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(batch, row, col, value);
      }
      ${
        this.outputShape[1] > 1 ?
            makeMatMulPackedSourceWgsl(
                [this.workPerThread, this.workPerThread, 1],
                this.workGroupSize) :
            makeMatMulVectorSourceWgsl(this.workGroupSize)}
    `;
    return userCode;
  }
}
