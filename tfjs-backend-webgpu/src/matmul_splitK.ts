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
import {getMainHeaderString} from './shader_preprocessor';
import {WebGPUProgram} from './webgpu_program';
import {computeDispatch} from './webgpu_util';

export function makeMatMulSplitKSource(): string {
  // atomicAdd only supports uint/int type. For float, we use
  // atomicCompareExchangeWeak to simulate.
  const atomicAddSnippet = `
     var assumed = atomicLoad(&(result.numbers[flatIndex]));
     var success = 0;
     for (; success == 0;) {
       let new = bitcast<f32>(assumed) + acc;
       let newI32 = bitcast<i32>(new);
       let resValue = atomicCompareExchangeWeak(&(result.numbers[flatIndex]), assumed, newI32);
       assumed = resValue[0];
       success = resValue[1];
     }
     `;
  return `
    var<workgroup> mm_Asub : array<array<f32, 16>, 16>;
    var<workgroup> mm_Bsub : array<array<f32, 16>, 16>;
    ${getMainHeaderString()}
      let batch = 0;
      let globalRow = i32(globalId.y);
      let globalCol = i32(globalId.x);
      let localRow = i32(localId.y);
      let localCol = i32(localId.x);
      let kStart = i32(globalId.z) * 64;
      var acc = 0.0;

      let numTiles = 4; // 64 / 16
      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t = t + 1) {
        // Load one tile of A into local memory.
        mm_Asub[localRow][localCol] = mm_readA(batch, globalRow,
            kStart + t * 16 + localCol);

        // Load one tile of B into local memory.
        mm_Bsub[localRow][localCol] = mm_readB(batch,
            kStart + t * 16 + localRow,
            globalCol);

        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < 16; k = k + 1) {
          acc = acc + mm_Asub[localRow][k] * mm_Bsub[k][localCol];
        }

        workgroupBarrier();
      }

      let coords = vec3<i32>(batch, globalRow, globalCol);
      let flatIndex = getOutputIndexFromCoords(coords);
      ${atomicAddSnippet}
    }
  `;
}

export class MatMulSplitKProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  uniforms = `dimAOuter : i32; dimBOuter : i32; dimInner : i32;`;
  workGroupSize: [number, number, number] = [16, 16, 1];
  transposeA: boolean;
  transposeB: boolean;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  atomic = true;

  constructor(
      outputShape: [number, number, number], dimInner: number,
      transposeA = false, transposeB = false, bias: TensorInfo = null,
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
        [1, 1, 64]);

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
    this.shaderKey =
        `matMulSplitK_${this.activation}_${transposeA}_${transposeB}`;
  }

  getUserCode(): string {
    let sampleA;
    if (this.transposeA === false) {
      sampleA =
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimAOuter, uniforms.dimInner))) {
            return A.numbers[batch * batchASize + row * uniforms.dimInner + col];
          }
          return 0.0;`;
    } else {
      sampleA =
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimAOuter, uniforms.dimInner))) {
            return A.numbers[batch* batchASize + col * uniforms.dimAOuter + row];
          }
          return 0.0;`;
    }

    let sampleB;
    if (this.transposeB === false) {
      sampleB =
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
            return B.numbers[batch * batchBSize + row * uniforms.dimBOuter + col];
          }
          return 0.0;`;
    } else {
      sampleB =
          `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
            return B.numbers[batch * batchBSize + col * uniforms.dimInner + row];
          }
          return 0.0;`;
    }

    const userCode = `
      fn mm_readA(batch: i32, row : i32, col : i32) -> f32 {
        let batchASize = uniforms.aShape[1] * uniforms.aShape[2];
        ${sampleA}
      }

      fn mm_readB(batch: i32, row : i32, col : i32) -> f32 {
        let batchBSize = uniforms.bShape[1] * uniforms.bShape[2];
        ${sampleB}
      }

      ${makeMatMulSplitKSource()}
    `;
    return userCode;
  }
}
