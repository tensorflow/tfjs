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
import {activationFnSnippet, biasActivationSnippet} from './activation_util';
import {getMainHeaderString as main, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

// No work group memory is used.
export class MatMulProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workGroupSize: [number, number, number] = [64, 1, 1];
  transposeA: boolean;
  transposeB: boolean;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  batchAEqualOne: boolean;
  batchBEqualOne: boolean;
  isVec4 = true;

  constructor(
      outputShape: [number, number, number], batchAEqualOne: boolean,
      batchBEqualOne: boolean, transposeA = false, transposeB = false,
      bias: TensorInfo = null, activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;

    this.dispatchLayout = flatDispatchLayout(this.outputShape);

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize, [4, 1, 1]);

    const addBias = bias != null;
    if (addBias) {
      this.variableNames.push('bias');
    }

    const hasPreluActivationWeights = preluActivationWeights != null;
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
    this.shaderKey = `matMul_${this.activation}_${transposeA}_${transposeB}_${
        this.batchAEqualOne}_${this.batchBEqualOne}`;
  }

  getUserCode(): string {
    const batchA = this.batchAEqualOne ? '0' : 'batch';
    const batchB = this.batchBEqualOne ? '0' : 'batch';
    const userCode = `
    ${
        activationFnSnippet(
            this.activation, this.hasPreluActivationWeights, this.isVec4)}
    const ColPerThread = 4;
    ${main('index')} {
      let flatIndex = index * ColPerThread;
      let coords = getCoordsFromIndex(flatIndex);
      let batch = coords.x;
      let globalRow = coords.y;
      let globalCol = coords.z;
      if (globalRow >= uniforms.dimAOuter || globalCol >= uniforms.dimBOuter)
      {
        return;
      }

      var value: vec4<f32>;
      // Loop over shared dimension.
      // Compute value values for a single thread.
      for (var k = 0; k < uniforms.dimInner; k = k + 4) {
        let BCached0 = getB(${batchB}, k, globalCol);
        let BCached1 = getB(${batchB}, k + 1, globalCol);
        let BCached2 = getB(${batchB}, k + 2, globalCol);
        let BCached3 = getB(${batchB}, k + 3, globalCol);

        let ACached = getA(${batchA}, globalRow, k);
        value = BCached0 * ACached.x + value;
        value = BCached1 * ACached.y + value;
        value = BCached2 * ACached.z + value;
        value = BCached3 * ACached.w + value;
      }
      ${biasActivationSnippet(this.addBias, this.activation)}
      setOutputAtCoords(coords[0], coords[1], coords[2], value);
    }
    `;
    return userCode;
  }
}
