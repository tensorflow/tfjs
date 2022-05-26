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

import {backend_util} from '@tensorflow/tfjs-core';

import {conv2dCommonSnippet} from './conv2d_common';
import {makeMatMulPackedSource} from './matmul_packed_webgpu';
import {WebGPUProgram} from './webgpu_program';
import {computeDispatch, computeWorkGroupSizeForConv2d, computeWorkPerThreadForConv2d} from './webgpu_util';

export class Conv2DMMProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms =
      `filterDims : vec2<i32>, pad : vec2<i32>, stride : vec2<i32>, dilation : vec2<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workGroupSize: [number, number, number];
  elementsPerThread: [number, number, number];
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  isChannelsLast: boolean;
  fitAOuter: boolean;
  fitBOuter: boolean;
  fitInner: boolean;
  tileInner: number;

  constructor(
      convInfo: backend_util.Conv2DInfo, dimAOuter: number, dimBOuter: number,
      dimInner: number, addBias = false,
      activation: backend_util.Activation = null,
      hasPreluActivationWeights = false) {
    this.outputShape = convInfo.outShape;
    this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
    this.dispatchLayout = this.isChannelsLast ? {x: [3], y: [1, 2], z: [0]} :
                                                {x: [2, 3], y: [1], z: [0]};
    this.workGroupSize =
        computeWorkGroupSizeForConv2d(this.dispatchLayout, this.outputShape);
    this.elementsPerThread =
        computeWorkPerThreadForConv2d(this.dispatchLayout, this.outputShape);

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        this.elementsPerThread);

    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;

    const tileAOuter = this.workGroupSize[1] * this.elementsPerThread[1];
    const tileBOuter = this.workGroupSize[0] * this.elementsPerThread[0];
    // TODO: Config this value based on dimInner/dimBOuter.
    this.tileInner = 32;

    this.fitAOuter = dimAOuter % tileAOuter === 0;
    this.fitBOuter = dimBOuter % tileBOuter === 0;
    this.fitInner = dimInner % this.tileInner === 0;

    this.shaderKey = `conv2DMM_${this.elementsPerThread}_${this.activation}}_${
        this.fitAOuter}_${this.fitBOuter}_${this.fitInner}_${
        this.isChannelsLast}`;
  }

  getUserCode(): string {
    const matMulSource = makeMatMulPackedSource(
        this.elementsPerThread, this.workGroupSize, !this.isChannelsLast,
        this.tileInner);
    const userCode = `
    ${
        conv2dCommonSnippet(
            this.isChannelsLast, this.fitAOuter, this.fitBOuter, this.fitInner,
            this.addBias, this.activation, this.hasPreluActivationWeights, 1, 1,
            1)}
    ${matMulSource}
  `;
    return userCode;
  }
}
