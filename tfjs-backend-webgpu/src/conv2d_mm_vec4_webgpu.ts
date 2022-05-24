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

import {mapActivationToShaderProgram} from './activation_util';
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

    this.shaderKey =
        `conv2DMMVec4_${this.activation}_${this.fitAOuter}_${this.fitBOuter}_${
            this.fitInner}_${this.elementsPerThread}_${this.innerElementSize}`;
  }

  getUserCode(): string {
    const matMulSource = makeMatMulPackedVec4Source(
        this.elementsPerThread, this.tileAOuter, this.tileBOuter,
        this.tileInner, this.innerElementSize, !this.isChannelsLast);
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
          c);
        ` :
                                                  `
        let outCoord = vec4<i32>(
          batch,
          row,
          c / outWidth,
          c % outWidth);
        `;
    const row = this.isChannelsLast ? 'r' : 'c';
    const col = this.isChannelsLast ? 'c' : 'r';
    const xHight =
        this.isChannelsLast ? 'uniforms.xShape[1]' : 'uniforms.xShape[2]';
    const xWidth =
        this.isChannelsLast ? 'uniforms.xShape[2]' : 'uniforms.xShape[3]';
    const readXSnippet = `
        let inChannels = uniforms.wShape[2];
        let outWidth = ${
        this.isChannelsLast ? 'uniforms.outShape[2]' : 'uniforms.outShape[3]'};
        let outRow = ${row} / outWidth;
        let outCol = ${row} % outWidth;
        let WRow = ${col} / (uniforms.filterDims[1] * inChannels);
        let WCol = ${col} / inChannels % uniforms.filterDims[1];
        let xCh = ${col} % inChannels;
        let xRow = outRow * uniforms.stride[0] + uniforms.dilation[0] * WRow - uniforms.pad[0];
        let xCol = outCol * uniforms.stride[1] + uniforms.dilation[1] * WCol - uniforms.pad[1];

        var resData = vec${this.innerElementSize}<f32>(0.0);
        // The bounds checking is always needed since we use it to pad zero for
        // the 'same' padding type.
        if (xRow >= 0 && xRow < ${xHight} && xCol >= 0 && xCol < ${xWidth}) {
          ${coordASnippet}
          let xIndex = getIndexFromCoords4D(coord, uniforms.xShape);
          ${
        this.innerElementSize === 3 ?
            'resData = vec3<f32>(x[xIndex], x[xIndex + 1], x[xIndex + 2]);' :
            'resData = x[xIndex / 4];'}
        }
        return resData;`;

    const sampleX = this.isChannelsLast ?
        (this.fitAOuter && this.fitInner ?
             `${readXSnippet}` :
             `if (row < uniforms.dimAOuter && c < uniforms.dimInner) {
      ${readXSnippet}
    }
    return vec${this.innerElementSize}<f32>(0.0);`) :
        (this.fitInner && this.fitBOuter ?
             `${readXSnippet}` :
             `if (row < uniforms.dimInner && c < uniforms.dimBOuter) {
          ${readXSnippet}
        }
        return vec${this.innerElementSize}<f32>(0.0);`);

    const sampleW = `return W[row * uniforms.wShape[3] / 4 + col];`;
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp =
          mapActivationToShaderProgram(this.activation, this.isVec4);
      if (this.hasPreluActivationWeights) {
        activationSnippet =
            `fn activation(a : vec4<f32>, outCoord : vec4<i32>) -> vec4<f32> {
          let b = getPreluActivationWeightsByOutputCoords(outCoord);
          ${activationOp}
        }`;
      } else {
        activationSnippet = `
        fn activation(a : vec4<f32>, outCoord : vec4<i32>) -> vec4<f32> {
          ${activationOp}
        }`;
      }

      applyActivationSnippet = `value = activation(value, outCoord);`;
    }

    const addBiasSnippet =
        this.addBias ? 'value = value + getBiasByOutputCoords(outCoord);' : '';

    const userCode = `
        ${activationSnippet}
        fn mm_readA(row : i32, col : i32, globalId : vec3<u32>) -> vec${
        this.innerElementSize}<f32> {
          let r = row;
          let c = col * ${this.innerElementSize};
          var batch = i32(globalId.z);
          ${this.isChannelsLast ? sampleX : sampleW}
        }

        fn mm_readB(row : i32, col : i32, globalId : vec3<u32>) -> vec4<f32> {
          let r = row;
          let c = col * ${this.innerElementSize};
          var batch = i32(globalId.z);
          ${this.isChannelsLast ? sampleW : sampleX}
        }

        fn mm_write(row : i32, col : i32, valueInput : vec4<f32>, globalId : vec3<u32>) {
          var batch = i32(globalId.z);
          var value = valueInput;
          var c = col * 4;
          if (row < uniforms.dimAOuter && c < uniforms.dimBOuter)
          {
            let outWidth = ${
        this.isChannelsLast ? 'uniforms.outShape[2]' : 'uniforms.outShape[3]'};
            ${coordResSnippet}
            ${addBiasSnippet}
            ${applyActivationSnippet}
            setOutputAtCoords(outCoord[0], outCoord[1], outCoord[2], outCoord[3],
              value);
          }
        }
        ${matMulSource}
      `;
    return userCode;
  }
}
