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
import {makeMatMulPackedVec4Source} from './matmul_packed_vec4_webgpu';
import {makeMatMulPackedSource} from './matmul_packed_webgpu';
import {WebGPUProgram} from './webgpu_program';
import {computeDispatch, computeWorkGroupSizeForConv2d, computeWorkPerThreadForConv2d} from './webgpu_util';

const typeSnippet = (innerElementSize: number) => {
  switch (innerElementSize) {
    case 1:
      return 'f32';
    case 2:
      return 'vec2<f32>';
    case 3:
      return 'vec3<f32>';
    case 4:
      return 'vec4<f32>';
    default:
      throw new Error(`innerElementSize ${innerElementSize} is not supported.`);
  }
};

function conv2dCommonSnippet(
    isChannelsLast: boolean, fitAOuter: boolean, fitBOuter: boolean,
    fitInner: boolean, addBias = false,
    activation: backend_util.Activation = null,
    hasPreluActivationWeights = false, innerElementSizeX = 4,
    innerElementSizeW = 4, innerElementSize = 4) {
  const getXSnippet = (innerElementSize: number) => {
    switch (innerElementSize) {
      case 1:
        return 'resData = x[xIndex];';
      case 3:
        return 'resData = vec3<f32>(x[xIndex], x[xIndex + 1], x[xIndex + 2]);';
      case 4:
        return 'resData = x[xIndex / 4];';
      default:
        throw new Error(
            `innerElementSize ${innerElementSize} is not supported.`);
    }
  };
  const getWSnippet = (innerElementSize: number) => {
    switch (innerElementSize) {
      case 1:
        return 'return W[row * uniforms.wShape[3] + colIn];';
      case 4:
        return 'return W[row * uniforms.wShape[3] / 4 + colIn];';
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
      let xRow = outRow * uniforms.stride[0] + uniforms.dilation[0] * WRow - uniforms.pad[0];
      let xCol = outCol * uniforms.stride[1] + uniforms.dilation[1] * WCol - uniforms.pad[1];
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
      let col = colIn * ${innerElementSizeX};
      ${readXSnippet}` :
                                                            `
      let col = colIn * ${innerElementSizeX};
      if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
        ${readXSnippet}
      }
      return ${typeSnippet(innerElementSizeX)}(0.0);`) :
                                   (fitInner && fitBOuter ? `
      let col = colIn * ${innerElementSizeX};
      ${readXSnippet}` :
                                                            `
      let col = colIn * ${innerElementSizeX};
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
  let activationSnippet = '', applyActivationSnippet = '';
  if (activation) {
    const activationOp =
        mapActivationToShaderProgram(activation, innerElementSize === 4);
    if (hasPreluActivationWeights) {
      activationSnippet =
          `fn activation(a: ${resType}, outCoord : vec4<i32>) -> ${resType} {
              let b = getPreluActivationWeightsByOutputCoords(outCoord);
              ${activationOp}
           }`;
    } else {
      activationSnippet = `
           fn activation(a : ${resType}, outCoord : vec4<i32>) -> ${resType} {
             ${activationOp}
           }`;
    }

    applyActivationSnippet = `value = activation(value, outCoord);`;
  }

  const addBiasSnippet =
      addBias ? 'value = value + getBiasByOutputCoords(outCoord);' : '';

  const userCode = `
      ${activationSnippet}
      fn mm_readA(row : i32, colIn : i32, globalId : vec3<u32>) -> ${aType} {
        let batch = i32(globalId.z);
        ${isChannelsLast ? sampleX : sampleW}
      }

      fn mm_readB(row : i32, colIn : i32, globalId : vec3<u32>) -> ${bType} {
        let batch = i32(globalId.z);
        ${isChannelsLast ? sampleW : sampleX}
      }

      fn mm_write(row : i32, colIn : i32, valueIn : ${
      resType}, globalId : vec3<u32>) {
        let col = colIn * ${innerElementSize};
        if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)
        {
        let batch = i32(globalId.z);
        var value = valueIn;
        let outWidth = ${
      isChannelsLast ? 'uniforms.outShape[2]' : 'uniforms.outShape[3]'};
        ${coordResSnippet}
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutputAtCoords(outCoord[0], outCoord[1], outCoord[2], outCoord[3], value);
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
  variableTypes: string[];
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
  tileAOuter: number;
  tileBOuter: number;
  tileInner: number;
  innerElementSize: number;
  isVec4?: boolean;

  constructor(
      convInfo: backend_util.Conv2DInfo, dimAOuter: number, dimBOuter: number,
      dimInner: number, addBias = false,
      activation: backend_util.Activation = null,
      hasPreluActivationWeights = false) {
    this.outputShape = convInfo.outShape;
    this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
    this.isVec4 =
        (((convInfo.inChannels % 4 === 0 || convInfo.inChannels % 3 === 0) &&
          this.isChannelsLast) ||
         (convInfo.outWidth % 4 === 0 && !this.isChannelsLast)) &&
        convInfo.outChannels % 4 === 0;
    this.dispatchLayout = this.isChannelsLast ? {x: [3], y: [1, 2], z: [0]} :
                                                {x: [2, 3], y: [1], z: [0]};
    this.workGroupSize = computeWorkGroupSizeForConv2d(
        this.dispatchLayout, this.outputShape, this.isVec4);
    this.elementsPerThread = computeWorkPerThreadForConv2d(
        this.dispatchLayout, this.outputShape, this.isVec4);

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        this.elementsPerThread);

    if (this.isVec4) {
      if (this.isChannelsLast && convInfo.inChannels % 4 !== 0) {
        this.innerElementSize = 3;
        this.variableTypes = ['f32', 'vec4<f32>'];
      } else {
        this.innerElementSize = 4;
        this.variableTypes = ['vec4<f32>', 'vec4<f32>'];
      }

      if (addBias) {
        this.variableNames.push('bias');
        this.variableTypes.push('vec4<f32>');
      }

      if (hasPreluActivationWeights) {
        this.variableNames.push('preluActivationWeights');
        this.variableTypes.push('vec4<f32>');
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

    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;

    this.tileAOuter = this.workGroupSize[1] * this.elementsPerThread[1];
    this.tileBOuter = this.workGroupSize[0] * this.elementsPerThread[0];
    this.tileInner = Math.max(
        this.workGroupSize[0] * this.innerElementSize, this.workGroupSize[1]);

    this.fitAOuter = dimAOuter % this.tileAOuter === 0;
    this.fitBOuter = dimBOuter % this.tileBOuter === 0;
    this.fitInner = dimInner % this.tileInner === 0;

    this.shaderKey = `conv2DMM_${this.elementsPerThread}_${this.activation}}_${
        this.fitAOuter}_${this.fitBOuter}_${this.fitInner}_${this.isVec4}_${
        this.innerElementSize}_${this.isChannelsLast}`;
  }

  getUserCode(): string {
    const matMulSource = this.isVec4 ?
        makeMatMulPackedVec4Source(
            this.elementsPerThread, this.tileAOuter, this.tileBOuter,
            this.tileInner, this.innerElementSize, !this.isChannelsLast) :
        makeMatMulPackedSource(
            this.elementsPerThread, this.workGroupSize, !this.isChannelsLast,
            this.tileInner);
    const elementsSize = this.isVec4 ?
        [this.isChannelsLast ? this.innerElementSize : 4, 4, 4] :
        [1, 1, 1];
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
