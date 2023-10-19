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

import {activationFnSnippet, biasActivationSnippet} from './activation_util';
import {makeMatMulPackedSource, makeMatMulPackedVec4Source} from './matmul_packed_webgpu';
import {typeSnippet, WebGPUProgram} from './webgpu_program';
import {computeDispatch, computeWorkgroupSizeForConv2d, computeWorkPerThreadForConv2d} from './webgpu_util';

function conv2dCommonSnippet(
    isChannelsLast: boolean, fitAOuter: boolean, fitBOuter: boolean,
    fitInner: boolean, addBias = false,
    activation: backend_util.Activation = null,
    hasPreluActivationWeights = false, innerElementSizeX = 4,
    innerElementSizeW = 4, innerElementSize = 4) {
  const getXSnippet = (innerElementSize: number) => {
    switch (innerElementSize) {
      case 1:
        return 'resData = f32(x[xIndex]);';
      case 3:
        return 'resData = vec3<f32>(x[xIndex], x[xIndex + 1], x[xIndex + 2]);';
      case 4:
        return 'resData = vec4<f32>(x[xIndex / 4]);';
      default:
        throw new Error(
            `innerElementSize ${innerElementSize} is not supported.`);
    }
  };
  const getWSnippet = (innerElementSize: number) => {
    switch (innerElementSize) {
      case 1:
        return 'return f32(W[row * uniforms.wShape[3] + col]);';
      case 4:
        return 'return vec4<f32>(W[(row * uniforms.wShape[3] + col) / 4]);';
      default:
        throw new Error(
            `innerElementSize ${innerElementSize} is not supported.`);
    }
  };
  const coordASnippet = isChannelsLast ? `
      let coord = vec4<i32>(batch, xRow, xCol, xCh);
      ` :
                                         `
      let coord = vec4<i32>(batch, xCh, xRow, xCol);
      `;

  const coordResSnippet = isChannelsLast ? `
      let coords = vec4<i32>(
        batch,
        row / outWidth,
        row % outWidth,
        col);
      ` :
                                           `
      let coords = vec4<i32>(
        batch,
        row,
        col / outWidth,
        col % outWidth);
      `;

  const xHight = isChannelsLast ? 'uniforms.xShape[1]' : 'uniforms.xShape[2]';
  const xWidth = isChannelsLast ? 'uniforms.xShape[2]' : 'uniforms.xShape[3]';
  const row = isChannelsLast ? 'row' : 'col';
  const col = isChannelsLast ? 'col' : 'row';
  const readXSnippet = `
      let inChannels = uniforms.wShape[2];
      let outWidth = ${
      isChannelsLast ? 'uniforms.outShape[2]' : 'uniforms.outShape[3]'};
      let outRow = ${row} / outWidth;
      let outCol = ${row} % outWidth;

      let WRow = ${col} / (uniforms.filterDims[1] * inChannels);
      let WCol = ${col} / inChannels % uniforms.filterDims[1];
      let xRow = outRow * uniforms.strides[0] + uniforms.dilations[0] * WRow - uniforms.pads[0];
      let xCol = outCol * uniforms.strides[1] + uniforms.dilations[1] * WCol - uniforms.pads[1];
      let xCh = ${col} % inChannels;
      var resData = ${typeSnippet(innerElementSizeX)}(0.0);
      // The bounds checking is always needed since we use it to pad zero for
      // the 'same' padding type.
      if (xRow >= 0 && xRow < ${xHight} && xCol >= 0 && xCol < ${xWidth}) {
        ${coordASnippet}
        let xIndex = getIndexFromCoords4D(coord, uniforms.xShape);
        ${getXSnippet(innerElementSizeX)}
      }
      return resData;`;

  const sampleX = isChannelsLast ? (fitAOuter && fitInner ? `
      ${readXSnippet}` :
                                                            `
      if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
        ${readXSnippet}
      }
      return ${typeSnippet(innerElementSizeX)}(0.0);`) :
                                   (fitInner && fitBOuter ? `
      ${readXSnippet}` :
                                                            `
      if (row < uniforms.dimInner && col < uniforms.dimBOuter) {
        ${readXSnippet}
      }
      return ${typeSnippet(innerElementSizeX)}(0.0);`);

  const sampleW = `${getWSnippet(innerElementSizeW)}`;

  const resType = typeSnippet(innerElementSize);
  const aType = isChannelsLast ? typeSnippet(innerElementSizeX) :
                                 typeSnippet(innerElementSizeW);
  const bType = isChannelsLast ? typeSnippet(innerElementSizeW) :
                                 typeSnippet(innerElementSizeX);
  const userCode = `
      ${
      activationFnSnippet(
          activation, hasPreluActivationWeights, innerElementSize === 4, 4)}
      fn mm_readA(batch: i32, row : i32, col : i32) -> ${aType} {
        ${isChannelsLast ? sampleX : sampleW}
      }

      fn mm_readB(batch: i32, row : i32, col : i32) -> ${bType} {
        ${isChannelsLast ? sampleW : sampleX}
      }

      fn mm_write(batch: i32, row : i32, col : i32, valueIn : ${resType}) {
        if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)
        {
        var value = valueIn;
        let outWidth = ${
      isChannelsLast ? 'uniforms.outShape[2]' : 'uniforms.outShape[3]'};
        ${coordResSnippet}
        ${biasActivationSnippet(addBias, activation)}
        setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
        }
      }`;
  return userCode;
}

export class Conv2DMMProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  variableComponents: number[];
  uniforms =
      `filterDims : vec2<i32>, pads : vec2<i32>, strides : vec2<i32>, dilations : vec2<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workgroupSize: [number, number, number];
  elementsPerThread: [number, number, number];
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  isChannelsLast: boolean;
  fitAOuter: boolean;
  fitBOuter: boolean;
  fitInner: boolean;
  tileAOuter: number;
  tileBOuter: number;
  tileInner: number;
  innerElementSize: number;
  isVec4?: boolean;
  outputComponent: number;
  private sequentialAccessByThreads: boolean;

  constructor(
      convInfo: backend_util.Conv2DInfo, dimAOuter: number, dimBOuter: number,
      dimInner: number, addBias = false,
      activation: backend_util.Activation = null,
      hasPreluActivationWeights = false, sequentialAccessByThreads = false) {
    this.outputShape = convInfo.outShape;
    this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
    this.isVec4 =
        (((convInfo.inChannels % 4 === 0 || convInfo.inChannels % 3 === 0) &&
          this.isChannelsLast) ||
         (convInfo.outWidth % 4 === 0 && !this.isChannelsLast)) &&
        convInfo.outChannels % 4 === 0;
    this.dispatchLayout = this.isChannelsLast ? {x: [3], y: [1, 2], z: [0]} :
                                                {x: [2, 3], y: [1], z: [0]};
    this.workgroupSize = computeWorkgroupSizeForConv2d(
        this.dispatchLayout, this.outputShape, this.isVec4);
    this.elementsPerThread = computeWorkPerThreadForConv2d(
        this.dispatchLayout, this.outputShape, this.isVec4);

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workgroupSize,
        this.elementsPerThread);

    if (this.isVec4) {
      this.outputComponent = 4;
      if (this.isChannelsLast && convInfo.inChannels % 4 !== 0) {
        this.innerElementSize = 3;
        this.variableComponents = [1, 4];
      } else {
        this.innerElementSize = 4;
        this.variableComponents = [4, 4];
      }

      if (addBias) {
        this.variableNames.push('bias');
        this.variableComponents.push(4);
      }

      if (hasPreluActivationWeights) {
        this.variableNames.push('preluActivationWeights');
        this.variableComponents.push(4);
      }
    } else {
      this.innerElementSize = this.elementsPerThread[0];
      if (addBias) {
        this.variableNames.push('bias');
      }

      if (hasPreluActivationWeights) {
        this.variableNames.push('preluActivationWeights');
      }
    }

    this.sequentialAccessByThreads = sequentialAccessByThreads;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;

    this.tileAOuter = this.workgroupSize[1] * this.elementsPerThread[1];
    this.tileBOuter = this.workgroupSize[0] * this.elementsPerThread[0];
    this.tileInner = Math.max(
        this.workgroupSize[0] * this.innerElementSize, this.workgroupSize[1]);

    this.fitAOuter = dimAOuter % this.tileAOuter === 0;
    this.fitBOuter = dimBOuter % this.tileBOuter === 0;
    this.fitInner = dimInner % this.tileInner === 0;

    this.shaderKey = `conv2DMM_${this.elementsPerThread}_${this.activation}}_${
        this.fitAOuter}_${this.fitBOuter}_${this.fitInner}_${this.isVec4}_${
        this.innerElementSize}_${this.isChannelsLast}_${
        this.sequentialAccessByThreads}`;
  }

  getUserCode(): string {
    const matMulSource = this.isVec4 ?
        makeMatMulPackedVec4Source(
            this.elementsPerThread, this.workgroupSize, !this.isChannelsLast,
            this.tileInner) :
        makeMatMulPackedSource(
            this.elementsPerThread, this.workgroupSize, !this.isChannelsLast,
            this.tileInner, false, null, this.sequentialAccessByThreads);
    const elementsSize =
        this.isVec4 ? [this.innerElementSize, 4, 4] : [1, 1, 1];
    const userCode = `
    ${
        conv2dCommonSnippet(
            this.isChannelsLast, this.fitAOuter, this.fitBOuter, this.fitInner,
            this.addBias, this.activation, this.hasPreluActivationWeights,
            elementsSize[0], elementsSize[1], elementsSize[2])}
    ${matMulSource}
  `;
    return userCode;
  }
}
