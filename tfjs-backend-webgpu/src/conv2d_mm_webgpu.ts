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

import {backend_util, util} from '@tensorflow/tfjs-core';

import {mapActivationToShaderProgram} from './activation_util';
import {makeMatMulPackedSource} from './matmul_packed_webgpu';
import {WebGPUProgram} from './webgpu_program';
import {computeDispatch, computeWorkGroupSizeForConv2d, computeWorkPerThreadForConv2d, tilesFitEvenlyIntoShape} from './webgpu_util';

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
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  fitA: boolean;
  fitB: boolean;
  isChannelsLast: boolean;

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: backend_util.Activation = null,
      hasPreluActivationWeights = false) {
    this.outputShape = convInfo.outShape;
    this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
    this.dispatchLayout = this.isChannelsLast ? {x: [3], y: [1, 2], z: [0]} :
                                                {x: [1], y: [2, 3], z: [0]};
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
    this.convInfo = convInfo;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;

    [this.fitA, this.fitB] = this.getShapeFit();
    this.shaderKey = `conv2DMM_${this.elementsPerThread}_${this.activation}_${
        this.fitA}_${this.fitB}_${this.isChannelsLast}`;
  }

  getShapeFit(): boolean[] {
    const tileAOuter = this.workGroupSize[1] * this.elementsPerThread[1];
    const tileBOuter = this.workGroupSize[0] * this.elementsPerThread[0];
    const tileInner = tileAOuter > tileBOuter ? tileAOuter : tileBOuter;
    util.assert(
        tileInner % this.workGroupSize[0] === 0 &&
            tileInner % this.workGroupSize[1] === 0,
        () =>
            // tslint:disable-next-line: max-line-length
        'tileInner must be multiple of workgroupsize.x and workgroupsize.y');
    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];
    const dimAOuter = this.convInfo.outHeight * this.convInfo.outWidth;
    const dimBOuter = this.convInfo.outChannels;
    const dimInner = this.convInfo.filterHeight * this.convInfo.filterWidth *
        this.convInfo.inChannels;

    return [
      tilesFitEvenlyIntoShape(tileSizeA, [dimAOuter, dimInner]),
      tilesFitEvenlyIntoShape(tileSizeB, [dimInner, dimBOuter])
    ];
  }

  getUserCode(): string {
    const coordASnippet = this.isChannelsLast ? `
    let coord = vec4<i32>(batch, xRow, xCol, col % inChannels);
    ` :
                                                `
    let coord = vec4<i32>(batch, col % inChannels, xRow, xCol);
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
      col,
      row / outWidth,
      row % outWidth);
    `;

    const matMulSource =
        makeMatMulPackedSource(this.elementsPerThread, this.workGroupSize);

    const readASnippet = `
    let inChannels = uniforms.wShape[2];
    let outWidth = ${
        this.isChannelsLast ? 'uniforms.outShape[2]' : 'uniforms.outShape[3]'};
    let outRow = row / outWidth;
    let outCol = row % outWidth;

    let WRow = col / (uniforms.filterDims[1] * inChannels);
    let WCol = col / inChannels % uniforms.filterDims[1];
    let xRow = outRow * uniforms.stride[0] + uniforms.dilation[0] * WRow - uniforms.pad[0];
    let xCol = outCol * uniforms.stride[1] + uniforms.dilation[1] * WCol - uniforms.pad[1];
    ${coordASnippet}
    // The bounds checking is always needed since we use it to pad zero for the
    // 'same' padding type.
    if(coordsInBounds4D(coord, uniforms.xShape)) {
      return x[getIndexFromCoords4D(coord, uniforms.xShape)];
    }
    return 0.0;`;

    const sampleA = this.fitA ?
        `${readASnippet}` :
        `if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
      ${readASnippet}
    }
    return 0.0;
    `;

    const sampleB = this.fitB ?
        `return W[row * uniforms.dimBOuter + col];` :
        `if(coordsInBounds2D(vec2<i32>(row, col), vec2<i32>(uniforms.dimInner, uniforms.dimBOuter))) {
           return W[row * uniforms.dimBOuter + col];
	 }
	 return 0.0;
	 `;

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
      ${sampleA}
    }

    fn mm_readB(row : i32, col : i32, globalId : vec3<u32>) -> f32 {
      ${sampleB}
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
