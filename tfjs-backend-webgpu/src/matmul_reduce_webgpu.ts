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
import {mapActivationToShaderProgram} from './activation_util';
import {getMainHeaderString, WebGPUProgram} from './webgpu_program';
import {computeDispatch} from './webgpu_util';

export function makeMatMulReduceSource(): string {
  return `
    var<workgroup> sumValues : array<f32, workGroupSizeX>;
    ${getMainHeaderString()}
      let coords = getOutputCoords();
      let batch = coords[0];
      let row = coords[1];
      let col = coords[2];
      var sum = 0.0;
      let Length = uniforms.dimInner;
      for (var k = i32(localId.x); k < Length; k = k + i32(workGroupSizeX)) {
        let dataA = mm_readA(batch, row, k);
        let dataB = mm_readB(batch, k, col);
        sum = sum + dataA * dataB;
      }
      sumValues[localId.x] = sum;
      workgroupBarrier();

      for(var currentSize = workGroupSizeX / 2u; currentSize > 1u;
          currentSize = currentSize / 2u) {
        if (localId.x < currentSize)
        {
          sumValues[localId.x] = sumValues[localId.x] + sumValues[localId.x + currentSize];
        }
        workgroupBarrier();
      }

      if (localId.x == 0u) {
        sum = sumValues[0] + sumValues[1];
        mm_write(batch, row, col, sum);
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
  uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workGroupSize: [number, number, number] = [256, 1, 1];
  transposeA: boolean;
  transposeB: boolean;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  batchAEqualOne: boolean;
  batchBEqualOne: boolean;

  constructor(
      outputShape: [number, number, number], batchAEqualOne: boolean,
      batchBEqualOne: boolean, transposeA = false, transposeB = false,
      bias: TensorInfo = null, activation: backend_util.Activation = null,
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
    this.batchAEqualOne = batchAEqualOne;
    this.batchBEqualOne = batchBEqualOne;
    this.shaderKey = `matMulReduce_${this.activation}_${transposeA}_${
        transposeB}_${this.batchAEqualOne}_${this.batchBEqualOne}`;
  }

  getUserCode(): string {
    let sampleA;
    if (this.transposeA === false) {
      sampleA =
          `return f32(A[batch * batchASize + row * uniforms.dimInner + col]);`;
    } else {
      sampleA =
          `return f32(A[batch * batchASize + col * uniforms.dimAOuter + row]);`;
    }

    let sampleB;
    if (this.transposeB === false) {
      sampleB =
          `return f32(B[batch * batchBSize + row * uniforms.dimBOuter + col]);`;
    } else {
      sampleB =
          `return f32(B[batch * batchBSize + col * uniforms.dimInner + row]);`;
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

      fn mm_readA(batchIn: i32, row : i32, col : i32) -> f32 {
        ${
        this.batchAEqualOne ? `
          let batchASize = 0;
          let batch = 0;
          ` :
                              `
          let batchASize = uniforms.aShape[1] * uniforms.aShape[2];
          let batch = batchIn;
          `}
        ${sampleA}
      }

      fn mm_readB(batchIn: i32, row : i32, col : i32) -> f32 {
        ${
        this.batchBEqualOne ? `
          let batch = 0;
          let batchBSize = 0;
          ` :
                              `
          let batch = batchIn;
          let batchBSize = uniforms.bShape[1] * uniforms.bShape[2];
          `}
        ${sampleB}
      }

      fn mm_write(batch: i32, row : i32, col : i32, valueIn : f32) {
        var value = valueIn;
        let outCoord = vec3<i32>(batch, row, col);
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutputAtCoords(batch, row, col, value);
      }
      ${makeMatMulReduceSource()}
    `;
    return userCode;
  }
}
