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

import {backend_util} from '@tensorflow/tfjs-core';

import {conv2dCommonSnippet} from './conv2d_common';
import {makeMatMulPackedVec4Source} from './matmul_packed_vec4_webgpu';
import {WebGPUProgram} from './webgpu_program';
import {computeDispatch} from './webgpu_util';

export class Conv2DMMVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  variableTypes: string[];
  uniforms =
      `filterDims : vec2<i32>, pad : vec2<i32>, stride : vec2<i32>, dilation : vec2<i32>,
      dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workGroupSize: [number, number, number] = [8, 8, 1];
  elementsPerThread: [number, number, number];
  isVec4 = true;
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  isChannelsLast: boolean;
  tileAOuter: number;
  tileBOuter: number;
  tileInner: number;
  fitAOuter: boolean;
  fitBOuter: boolean;
  fitInner: boolean;
  innerElementSize: number;

  constructor(
      convInfo: backend_util.Conv2DInfo, dimAOuter: number, dimBOuter: number,
      dimInner: number, addBias = false,
      activation: backend_util.Activation = null,
      hasPreluActivationWeights = false) {
    this.outputShape = convInfo.outShape;
    this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
    this.dispatchLayout = this.isChannelsLast ? {x: [3], y: [1, 2], z: [0]} :
                                                {x: [2, 3], y: [1], z: [0]};
    // The first element in elementsPerThread must be 4.
    this.elementsPerThread = [4, 4, 1];
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        this.elementsPerThread);
    this.convInfo = convInfo;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.innerElementSize = this.convInfo.inChannels % 4 === 0 ? 4 : 3;
    if (this.innerElementSize === 3) {
      this.variableTypes = ['f32', 'vec4<f32>'];
    } else {
      this.variableTypes = ['vec4<f32>', 'vec4<f32>'];
    }

    if (this.addBias) {
      this.variableNames.push('bias');
      this.variableTypes.push('vec4<f32>');
    }

    if (this.hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
      this.variableTypes.push('vec4<f32>');
    }
    this.tileAOuter = this.workGroupSize[1] * this.elementsPerThread[1];
    this.tileBOuter = this.workGroupSize[0] * this.elementsPerThread[0];
    this.tileInner = this.workGroupSize[0] * this.innerElementSize;
    this.fitAOuter = dimAOuter % this.tileAOuter === 0;
    this.fitBOuter = dimBOuter % this.tileBOuter === 0;
    this.fitInner = dimInner % this.tileInner === 0;

    this.shaderKey = `conv2DMMVec4_${this.activation}_${this.fitAOuter}_${
        this.fitBOuter}_${this.fitInner}_${this.elementsPerThread}_${
        this.innerElementSize}_${this.isChannelsLast}`;
  }

  getUserCode(): string {
    const matMulSource = makeMatMulPackedVec4Source(
        this.elementsPerThread, this.tileAOuter, this.tileBOuter,
        this.tileInner, this.innerElementSize, !this.isChannelsLast);

    const userCode = `
    ${
        conv2dCommonSnippet(
            this.isChannelsLast, this.fitAOuter, this.fitBOuter, this.fitInner,
            this.addBias, this.activation, this.hasPreluActivationWeights,
            this.isChannelsLast ? this.innerElementSize : 4, 4, 4)}
        ${matMulSource}
      `;
    return userCode;
  }
}
