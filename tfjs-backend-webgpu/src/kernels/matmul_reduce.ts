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

import {backend_util, TensorInfo} from '@tensorflow/tfjs-core';

import {getGlobalIndexString, getMainHeaderString} from '../shader_preprocessor';
import {computeDispatch} from '../webgpu_util';

import {mapActivationToShaderProgram} from './activation_util';
import {WebGPUProgram} from './webgpu_program';

export function makeMatMulReduceSource(): string {
  return `
    fn DIV_CEIL(a : i32, b : i32) -> i32 {
      return ((a - 1) / b + 1);
    }

    var<workgroup> sumValues : array<f32, workGroupSizeX>;
    ${getMainHeaderString()} {
      ${getGlobalIndexString()}
      let coords = getOutputCoords(globalId, index);
      let batch = coords[0];
      let row = coords[1];
      let col = coords[2];
      var bestValue = 0.0;
      let Length = uniforms.dimInner;
      let WorkPerThread = DIV_CEIL(Length, i32(workGroupSizeX));
      for (var w = 0; w < WorkPerThread; w = w + 1) {
        let i = i32(localId.x) * WorkPerThread + w;
        if (i < Length) {
          let candidate = mm_read(batch, row, col, i);
          bestValue = bestValue + candidate;
        }
      }
      sumValues[localId.x] = bestValue;

      bestValue = 0.0;
      var currentSize = i32(workGroupSizeX);
      for(; currentSize > 1;) {
        workgroupBarrier();
        for (var w = 0; w < 2; w = w + 1) {
          let i = i32(localId.x) * 2 + w;
          if (i < currentSize) {
            let candidate = sumValues[i];
            bestValue = bestValue + candidate;
          }
        }
        workgroupBarrier();
        sumValues[localId.x] = bestValue;
        currentSize = DIV_CEIL(currentSize, 2);
        if(currentSize > 1) { bestValue = 0.0; }
      }
      if (localId.x == 0u) {
        mm_write(batch, row, col, bestValue);
      }
    }
  `;
}

export class MatMulReduceProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  uniforms = `dimAOuter : i32; dimBOuter : i32; dimInner : i32;`;
  workGroupSize: [number, number, number] = [256, 1, 1];
  transposeA: boolean;
  transposeB: boolean;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;

  constructor(
      outputShape: [number, number, number], transposeA = false,
      transposeB = false, bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [], y: [1, 2], z: [0]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

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
        `matMulReduce_${this.activation}_${transposeA}_${transposeB}`;
  }

  getUserCode(): string {
    let sampleA;
    if (this.transposeA === false) {
      sampleA =
          `return A.numbers[batch * batchASize + row * uniforms.dimInner + col];`;
    } else {
      sampleA =
          `return A.numbers[batch * batchASize + col * uniforms.dimAOuter + row];`;
    }

    let sampleB;
    if (this.transposeB === false) {
      sampleB =
          `return B.numbers[batch * batchBSize + row * uniforms.dimBOuter + col];`;
    } else {
      sampleB =
          `return B.numbers[batch * batchBSize + col * uniforms.dimInner + row];`;
    }

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(this.activation, false);
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

      fn mm_readA(batch: i32, row : i32, col : i32) -> f32 {
        let batchASize = uniforms.aShape[1] * uniforms.aShape[2];
        ${sampleA}
      }

      fn mm_readB(batch: i32, row : i32, col : i32) -> f32 {
        let batchBSize = uniforms.bShape[1] * uniforms.bShape[2];
        ${sampleB}
      }

      fn mm_read(batch: i32, row : i32, col : i32, k: i32) -> f32 {
        let dataA = mm_readA(batch, row, k);
        let dataB = mm_readB(batch, k, col);
        return dataA * dataB;
      }

      fn mm_write(batch: i32, row : i32, col : i32, valueIn : f32) {
        var value = valueIn;
        let outCoord = vec3<i32>(batch, row, col);
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(batch, row, col, value);
      }
      ${makeMatMulReduceSource()}
    `;
    return userCode;
  }
}
