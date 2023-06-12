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

import {backend_util, TensorInfo, util} from '@tensorflow/tfjs-core';

import {activationFnSnippet, biasActivationSnippet} from './activation_util';
import {makeMatMulPackedSource, makeMatMulPackedVec4Source, matMulReadFnSource} from './matmul_packed_webgpu';
import {atomicAddSnippet} from './shader_util';
import {getMainHeaderString as main, typeSnippet, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class MatMulSplitKProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workgroupSize: [number, number, number] = [8, 8, 1];
  elementsPerThread: [number, number, number];
  transposeA: boolean;
  transposeB: boolean;
  atomic = true;
  outputComponent: number;
  splitedDimInner = 128;

  constructor(
      outputShape: [number, number, number], dimInner: number,
      transposeA = false, transposeB = false) {
    util.assert(
        outputShape[0] === 1,
        () => 'MatMulSplitKProgram only supports batch = 1.');
    this.outputShape = outputShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0, 3]};
    const isVec4 = (transposeA && this.outputShape[1] % 4 === 0 ||
                    !transposeA && dimInner % 4 === 0) &&
        this.outputShape[2] % 4 === 0;
    this.elementsPerThread = [4, 4, this.splitedDimInner];
    this.outputComponent = isVec4 ? 4 : 1;
    if (!isVec4) {
      if (this.outputShape[1] < 16) {
        this.elementsPerThread[1] = 1;
      }
      if (this.outputShape[2] < 16) {
        this.elementsPerThread[0] = 1;
      }
    }

    this.dispatch = computeDispatch(
        this.dispatchLayout,
        [
          this.outputShape[0], this.outputShape[1], this.outputShape[2],
          dimInner
        ],
        this.workgroupSize, this.elementsPerThread);

    this.transposeA = transposeA;
    this.transposeB = transposeB;
    this.shaderKey = `matMulSplitK_${transposeA}_${transposeB}_${
        this.elementsPerThread}_${this.outputComponent}`;
  }

  getUserCode(): string {
    const component = this.outputComponent;
    const userCode = `
      ${
        matMulReadFnSource(
            false, this.transposeB, false, false, false, component)}
      fn mm_write(batch: i32, row : i32, col : i32, value : ${
        typeSnippet(component)}) {
        if (row < uniforms.dimAOuter && col < uniforms.dimBOuter) {
          let coords = vec3<i32>(batch, row, col);
          let flatIndex = getOutputIndexFromCoords(coords);
          // The problem is that we should initialize output to zero before using.
          // Otherwise, the original value will be added to the result.
          for (var i = 0; i < ${component}; i = i + 1) {
            ${
        atomicAddSnippet(
            '&result[flatIndex + i]', `${component > 1 ? 'value[i]' : 'value'}`,
            'float32')}
          }
        }
      }
      ${
        component === 4 ? makeMatMulPackedVec4Source(
                              this.elementsPerThread, this.workgroupSize,
                              this.transposeA, 32, true, this.splitedDimInner) :
                          makeMatMulPackedSource(
                              this.elementsPerThread, this.workgroupSize,
                              this.transposeA, 32, true, this.splitedDimInner)}
    `;
    return userCode;
  }
}

export class BiasActivationProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  uniforms = '';
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x'];
  workgroupSize: [number, number, number] = [64, 1, 1];
  size = true;
  private addBias: boolean;
  private activation: backend_util.Activation;
  private hasPreluActivationWeights: boolean;

  constructor(
      outputShape: number[], bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize);
    this.addBias = bias != null;
    this.hasPreluActivationWeights = preluActivationWeights != null;
    this.activation = activation;
    if (this.addBias) {
      this.variableNames.push('bias');
    }

    if (this.hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.shaderKey = `biasActivation_${activation}`;
  }

  getUserCode(): string {
    return `
    ${activationFnSnippet(this.activation, this.hasPreluActivationWeights)}
    ${main('index')} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        var value = getXByOutputIndex(index);
        ${biasActivationSnippet(this.addBias, this.activation)}
        setOutputAtIndex(index, value);
      }
    }
    `;
  }
}
