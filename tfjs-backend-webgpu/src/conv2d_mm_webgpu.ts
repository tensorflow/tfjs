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

import {mapActivationToShaderProgram} from './activation_util';
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
    const coordASnippet = this.isChannelsLast ? `
    let coord = vec4<i32>(batch, xRow, xCol, xCh);
    ` :
                                                `
    let coord = vec4<i32>(batch, xCh, xRow, xCol);
    `;

    const coordResSnippet = this.isChannelsLast ? `
    let outCoord = vec4<i32>(
      batch,
      row / outWidth,
      row % outWidth,
      col);
    ` :
                                                  `
    let outCoord = vec4<i32>(
      batch,
      row,
      col / outWidth,
      col % outWidth);
    `;

    const matMulSource = makeMatMulPackedSource(
        this.elementsPerThread, this.workGroupSize, !this.isChannelsLast,
        this.tileInner);

    const row = this.isChannelsLast ? 'row' : 'col';
    const col = this.isChannelsLast ? 'col' : 'row';
    const readXSnippet = `
    let inChannels = uniforms.wShape[2];
    let outWidth = ${
        this.isChannelsLast ? 'uniforms.outShape[2]' : 'uniforms.outShape[3]'};
    let outRow = ${row} / outWidth;
    let outCol = ${row} % outWidth;

    let WRow = ${col} / (uniforms.filterDims[1] * inChannels);
    let WCol = ${col} / inChannels % uniforms.filterDims[1];
    let xRow = outRow * uniforms.stride[0] + uniforms.dilation[0] * WRow - uniforms.pad[0];
    let xCol = outCol * uniforms.stride[1] + uniforms.dilation[1] * WCol - uniforms.pad[1];
    let xCh = ${col} % inChannels;
    ${coordASnippet}
    // The bounds checking is always needed since we use it to pad zero for the
    // 'same' padding type.
    if(coordsInBounds4D(coord, uniforms.xShape)) {
      return x[getIndexFromCoords4D(coord, uniforms.xShape)];
    }
    return 0.0;`;

    const sampleX = this.isChannelsLast ?
        (this.fitAOuter && this.fitInner ?
             `${readXSnippet}` :
             `if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
      ${readXSnippet}
    }
    return 0.0;`) :
        (this.fitInner && this.fitBOuter ?
             `${readXSnippet}` :
             `if (row < uniforms.dimInner && col < uniforms.dimBOuter) {
          ${readXSnippet}
        }
        return 0.0;`);

    const sampleW = `return W[row * uniforms.wShape[3] + col];`;

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(this.activation, false);
      if (this.hasPreluActivationWeights) {
        activationSnippet =
            `fn activation(a: f32, outCoord : vec4<i32>) -> f32 {
                  let b = getPreluActivationWeightsByOutputCoords(outCoord);
                  ${activationOp}
                }`;
      } else {
        activationSnippet = `
                  fn activation(a : f32, outCoord : vec4<i32>) -> f32 {
                    ${activationOp}
                  }
                `;
      }

      applyActivationSnippet = `value = activation(value, outCoord);`;
    }

    const addBiasSnippet =
        this.addBias ? 'value = value + getBiasByOutputCoords(outCoord);' : '';

    const userCode = `
    ${activationSnippet}
    fn mm_readA(row : i32, col : i32, globalId : vec3<u32>) -> f32 {
      var batch = i32(globalId.z);
      ${this.isChannelsLast ? sampleX : sampleW}
    }

    fn mm_readB(row : i32, col : i32, globalId : vec3<u32>) -> f32 {
      var batch = i32(globalId.z);
      ${this.isChannelsLast ? sampleW : sampleX}
    }

    fn mm_write(row : i32, col : i32, valueInput : f32, globalId : vec3<u32>) {
      var batch = i32(globalId.z);
      var value = valueInput;
      let outWidth = ${
        this.isChannelsLast ? 'uniforms.outShape[2]' : 'uniforms.outShape[3]'};
      ${coordResSnippet}
      ${addBiasSnippet}
      ${applyActivationSnippet}
      result[getIndexFromCoords4D(outCoord, uniforms.outShape)] = value;
    }
    ${matMulSource}
  `;
    return userCode;
  }
}
