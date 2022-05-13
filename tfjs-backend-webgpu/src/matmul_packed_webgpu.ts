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

import {mapActivationToShaderProgram} from './activation_util';
import {getMainHeaderString} from './shader_preprocessor';
import {WebGPUProgram} from './webgpu_program';
import {computeDispatch} from './webgpu_util';

const storeDataToSubASnippet =
    (transpose: boolean) => {
      if (transpose) {
        return `
        mm_Asub[inputRow][inputCol] = mm_readA(
          t * TileInner + inputRow,
          globalRowStart + inputCol, globalId);
        `;

      } else {
        return `
        mm_Asub[inputRow][inputCol] = mm_readA(
          globalRowStart + inputRow,
          t * TileInner + inputCol, globalId);
        `;
      }
    }

const loadDataFromSubASnippet =
    (transposeA: boolean) => {
      return transposeA ? 'let ACached = mm_Asub[k][tileRow + innerRow];' :

                          'let ACached = mm_Asub[tileRow + innerRow][k];';
    }

export function
makeMatMulPackedSource(
    workPerThread: number[], workGroupSize: [number, number, number],
    transposeA = false):
    string {
      util.assert(
          workGroupSize[0] === workGroupSize[1] &&
              workPerThread[0] === workPerThread[1],
          () => 'The workGroupSize and workPerThread must be square.');
      const tileInner = workGroupSize[0] * workPerThread[0];
      return `
    var<workgroup> mm_Asub : array<array<f32, ${tileInner}>, ${tileInner}>;
    var<workgroup> mm_Bsub : array<array<f32, ${tileInner}>, ${tileInner}>;
    let RowPerThread = ${workPerThread[1]};
    let ColPerThread = ${workPerThread[0]};
    let TileInner = ${tileInner};

    @stage(compute) @workgroup_size(workGroupSizeX, workGroupSizeY, workGroupSizeZ)
    fn main(@builtin(local_invocation_id) LocalId : vec3<u32>,
            @builtin(global_invocation_id) GlobalId : vec3<u32>,
            @builtin(num_workgroups) NumWorkgroups: vec3<u32>,
            @builtin(workgroup_id) workgroupId: vec3<u32>) {
      localId = LocalId;
      globalId = GlobalId;
      numWorkgroups = NumWorkgroups;

      let tileRow = i32(localId.y) * RowPerThread;
      let tileCol = i32(localId.x) * ColPerThread;

      let globalRow = i32(globalId.y) * RowPerThread;
      let globalCol = i32(globalId.x) * ColPerThread;

      let globalRowStart = i32(workgroupId.y) * TileInner;
      let globalColStart = i32(workgroupId.x) * TileInner;

      let numTiles = (uniforms.dimInner - 1) / TileInner + 1;

      var acc : array<array<f32, ColPerThread>, RowPerThread>;

      // Without this initialization strange values show up in acc.
      for (var innerRow = 0; innerRow < RowPerThread; innerRow = innerRow + 1) {
        for (var innerCol = 0; innerCol < ColPerThread; innerCol = innerCol + 1) {
          acc[innerRow][innerCol] = 0.0;
        }
      }

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t = t + 1) {
        // Load one tile of A into local memory.
        for (var inputRow = i32(localId.y); inputRow < TileInner; inputRow = inputRow + i32(workGroupSizeY)) {
         for (var inputCol = i32(localId.x); inputCol < TileInner; inputCol = inputCol + i32(workGroupSizeX)) {
            ${storeDataToSubASnippet(transposeA)}
            mm_Bsub[inputRow][inputCol] = mm_readB(
              t * TileInner + inputRow,
              globalColStart + inputCol, globalId);
          }
        }

        workgroupBarrier();

        // Compute acc values for a single thread.
        var BCached : array<f32, ColPerThread>;
        for (var k = 0; k < TileInner; k = k + 1) {
          for (var inner = 0; inner < ColPerThread; inner = inner + 1) {
            BCached[inner] = mm_Bsub[k][tileCol + inner];
          }

          for (var innerRow = 0; innerRow < RowPerThread; innerRow = innerRow + 1) {
            ${loadDataFromSubASnippet(transposeA)}
            for (var innerCol = 0; innerCol < ColPerThread; innerCol = innerCol + 1) {
              acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];
            }
          }
        }

        workgroupBarrier();
      }

      for (var innerRow = 0; innerRow < RowPerThread; innerRow = innerRow + 1) {
        for (var innerCol = 0; innerCol < ColPerThread; innerCol = innerCol + 1) {

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

export function
makeMatMulVectorSource(workGroupSize: [number, number, number]):
    string {
      return `
    let TileSize = ${workGroupSize[0] * 4};
    var<workgroup> mm_Asub : array<vec4<f32>, ${workGroupSize[0]}>;

    ${getMainHeaderString()}
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
  uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workGroupSize: [number, number, number] = [8, 8, 1];
  transposeA: boolean;
  transposeB: boolean;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  batchAEqualOne: boolean;
  batchBEqualOne: boolean;

  constructor(
      aShape: [number, number, number], outputShape: [number, number, number],
      workPerThread: number, batchAEqualOne: boolean, batchBEqualOne: boolean,
      transposeA = false, transposeB = false, bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0]};
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
    this.transposeA = transposeA;
    this.transposeB = transposeB;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.batchAEqualOne = batchAEqualOne;
    this.batchBEqualOne = batchBEqualOne;

    this.shaderKey = `matMulPacked_${this.workPerThread}_${transposeA}_${
        transposeB}_${this.activation}_${this.outputShape[1] > 1}_${
        this.batchAEqualOne}_${this.batchBEqualOne}`;
  }

  getUserCode(): string {
    let sampleB;
    if (this.transposeB === false) {
      sampleB =
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
                 return B[batch * batchBSize + row * uniforms.dimBOuter + col];
               }
               return 0.0;`;
    } else {
      sampleB =
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
                 return B[batch * batchBSize + col * uniforms.dimInner + row];
               }
               return 0.0;`;
    }
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(this.activation, false);
      if (this.hasPreluActivationWeights) {
        activationSnippet =
            `fn activation(a : f32, outCoord : vec3<i32>) -> f32 {
               let b = getPreluActivationWeightsByOutputCoords(outCoord);
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

    const addBiasSnippet =
        this.addBias ? 'value = value + getBiasByOutputCoords(outCoord);' : '';

    const userCode = `
      ${activationSnippet}

      fn mm_readA(row : i32, col : i32,  globalId : vec3<u32>) -> f32 {
        ${
        this.batchAEqualOne ? `
        let batch = 0;
        ` :
                              `
        let batch = i32(globalId.z);
        `}
        if(coordsInBounds3D(vec3<i32>(batch, row, col), uniforms.aShape)) {
          return getA(batch, row, col);
        }
        return 0.0;
      }

      fn mm_readB(row : i32, col : i32,  globalId : vec3<u32>) -> f32 {
        ${
        this.batchBEqualOne ? `
        let batch = 0;
        let batchBSize = 0;
        ` :
                              `
        let batch = i32(globalId.z);
        let batchBSize = uniforms.bShape[1] * uniforms.bShape[2];
        `}
        ${sampleB}
      }

      fn mm_write(row : i32, col : i32, valueIn : f32, globalId : vec3<u32>) {
        var value = valueIn;
        let batch = i32(globalId.z);
        let outCoord = vec3<i32>(batch, row, col);
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutputAtCoords(batch, row, col, value);
      }
      ${
        makeMatMulPackedSource(
            [this.workPerThread, this.workPerThread, 1], this.workGroupSize,
            this.transposeA)}
    `;
    return userCode;
  }
}
