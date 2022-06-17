/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

// import {mapActivationToShaderProgram} from './activation_util';
import {getMainHeaderString, WebGPUProgram} from './webgpu_program';
import {computeDispatch} from './webgpu_util';

export class MatMulSplitKProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workGroupSize: [number, number, number] = [8, 8, 1];
  transposeA: boolean;
  transposeB: boolean;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  atomic = true;
  batchAEqualOne: boolean;
  batchBEqualOne: boolean;

  constructor(
      outputShape: [number, number, number], dimInner: number,
      batchAEqualOne: boolean, batchBEqualOne: boolean, transposeA = false,
      transposeB = false, bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0, 3]};
    this.dispatch = computeDispatch(
        this.dispatchLayout,
        [
          this.outputShape[0], this.outputShape[1], this.outputShape[2],
          dimInner
        ],
        this.workGroupSize, [4, 4, 32]);

    const addBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.transposeA = transposeA;
    this.transposeB = transposeB;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.batchAEqualOne = batchAEqualOne;
    this.batchBEqualOne = batchBEqualOne;
    this.shaderKey = `matMulSplitK_${this.activation}_${transposeA}_${
        transposeB}_${batchAEqualOne}_${batchBEqualOne}`;
  }

  getUserCode(): string {
    let sampleA;
    if (this.transposeA === false) {
      sampleA =
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimAOuter, uniforms.dimInner))) {
            return A[batch * batchASize + row * uniforms.dimInner + col];
          }
          return 0.0;`;
    } else {
      sampleA =
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimAOuter, uniforms.dimInner))) {
            return A[batch* batchASize + col * uniforms.dimAOuter + row];
          }
          return 0.0;`;
    }

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

    // atomicAdd only supports uint/int type. For float, we use
    // atomicCompareExchangeWeak to simulate.
    const atomicAddSnippet = `
     var oldValue = atomicLoad(&(result[flatIndex]));
     var exchanged = false;
     for (; !exchanged;) {
       let newValueF32 = bitcast<f32>(oldValue) + value;
       let newValue = bitcast<i32>(newValueF32);
       let res = atomicCompareExchangeWeak(&(result[flatIndex]), oldValue, newValue);
       oldValue = res.old_value;
       exchanged = res.exchanged;
     }
     `;
    const userCode = `
      fn mm_readA(batch: i32, row : i32, col : i32) -> f32 {
        let batchASize = uniforms.aShape[1] * uniforms.aShape[2];
        ${sampleA}
      }

      fn mm_readB(batch: i32, row : i32, col : i32) -> f32 {
        let batchBSize = uniforms.bShape[1] * uniforms.bShape[2];
        ${sampleB}
      }

      fn mm_write(batch: i32, row : i32, col : i32, valueIn : f32) {
        if (row < uniforms.dimAOuter && col < uniforms.dimBOuter) {
          let coords = vec3<i32>(batch, row, col);
          let flatIndex = getOutputIndexFromCoords(coords);
          var value = valueIn;
         // The problem is that we should initialize output to zero before using.
         // Otherwise, the original value will be added to the result.
         ${atomicAddSnippet}
        }
      }

      ${this.makeMatMulSplitKSource()}
    `;
    return userCode;
  }

  makeMatMulSplitKSource(): string {
    return `
      var<workgroup> mm_Asub : array<array<f32, 32>, 32>;
      var<workgroup> mm_Bsub : array<array<f32, 32>, 32>;
      ${getMainHeaderString()}
        let batch = 0;
        let tileRow = i32(localId.y) * 4;
        let tileCol = i32(localId.x) * 4;

        let globalRow = i32(globalId.y) * 4;
        let globalCol = i32(globalId.x) * 4;
        let kStart = i32(globalId.z) * 32;

        var acc : array<array<f32, 4>, 4>;
        var ACached : f32;
        var BCached : array<f32, 4>;

        // Without this initialization strange values show up in acc.
        for (var innerRow = 0; innerRow < 4; innerRow = innerRow + 1) {
          for (var innerCol = 0; innerCol < 4; innerCol = innerCol + 1) {
            acc[innerRow][innerCol] = 0.0;
          }
        }

        // Loop over shared dimension.
          // Load one tile of A and B into local memory.
          for (var innerRow = 0; innerRow < 4; innerRow = innerRow + 1) {
            for (var innerCol = 0; innerCol < 4; innerCol = innerCol + 1) {
              let inputRow = tileRow + innerRow;
              let inputCol = tileCol + innerCol;

              mm_Asub[inputRow][inputCol] = mm_readA(${
        this.batchAEqualOne ? 0 : 'batch'},
                  globalRow + innerRow,
                  kStart + inputCol);
              mm_Bsub[inputRow][inputCol] = mm_readB(${
        this.batchBEqualOne ? 0 : 'batch'},
                  kStart + inputRow,
                  globalCol + innerCol);
            }
          }

          workgroupBarrier();

          // Compute acc values for a single thread.
          for (var k = 0; k < 32; k = k + 1) {
            for (var inner = 0; inner < 4; inner = inner + 1) {
              BCached[inner] = mm_Bsub[k][tileCol + inner];
            }

            for (var innerRow = 0; innerRow < 4; innerRow = innerRow + 1) {
              ACached = mm_Asub[tileRow + innerRow][k];
              for (var innerCol = 0; innerCol < 4; innerCol = innerCol + 1) {
                acc[innerRow][innerCol] = acc[innerRow][innerCol] + ACached * BCached[innerCol];
              }
            }
          }

            for (var innerRow = 0; innerRow < 4; innerRow = innerRow + 1) {
              for (var innerCol = 0; innerCol < 4; innerCol = innerCol + 1) {
                mm_write(batch, globalRow + innerRow, globalCol + innerCol, acc[innerRow][innerCol]);
              }
            }
      }
    `;
  }
}
