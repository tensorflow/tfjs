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

import {getShapeCoords} from '../shader_preprocessor';
import {computeDispatch, computeWorkGroupSizeForConv2d, computeWorkPerThreadForConv2d, tilesFitEvenlyIntoShape} from '../webgpu_util';

import {makeMatMulPackedSource} from './matmul_packed_webgpu';
import {WebGPUProgram} from './webgpu_program';

export class Conv2DMMProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation;';
  workGroupSize: [number, number, number];
  elementsPerThread: [number, number, number];
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: string;
  hasPreluActivationWeights: boolean;

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: string = null, hasPreluActivationWeights = false) {
    this.outputShape = convInfo.outShape;

    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');
    this.dispatchLayout = {x: [3], y: [1, 2], z: [0]};
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
    this.shaderKey = `conv2DMM_${this.elementsPerThread}_${this.activation}`;
  }

  getUserCode(): string {
    const matMulSource = makeMatMulPackedSource(this.elementsPerThread);

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
    const dimAOuter = this.outputShape[1] * this.outputShape[2];
    const dimBOuter = this.outputShape[3];
    const dimInner = this.convInfo.filterHeight * this.convInfo.filterWidth *
        this.convInfo.inChannels;

    const readASnippet = `
    int outRow = row / ${this.outputShape[2]};
    int outCol = row % ${this.outputShape[2]};

    int WRow = col / (filterDims[1] * ${this.convInfo.inShape[3]});
    int WCol = (col / ${this.convInfo.inShape[3]}) % filterDims[1];

    ivec4 coord = ivec4(
        batch,
        outRow * stride[0] + dilation[0] * WRow - pad[0],
        outCol * stride[1] + dilation[1] * WCol - pad[1],
        col % ${this.convInfo.inShape[3]});
    // The bounds checking is always needed since we use it to pad zero for the
    // 'same' padding type.
    return coordsInBounds(coord, ${
        getShapeCoords(this.convInfo.inShape)}) ? x[getFlatIndex(coord, ${
        getShapeCoords(this.convInfo.inShape)})] : 0;`;

    const fitA = tilesFitEvenlyIntoShape(tileSizeA, [dimAOuter, dimInner]);
    const sampleA =
        fitA ? `${readASnippet}` : `if (row < dimAOuter && col < dimInner) {
      ${readASnippet}
    } else {
      return 0;
    }`;

    const fitB = tilesFitEvenlyIntoShape(tileSizeB, [dimInner, dimBOuter]);
    const sampleB = fitB ?
        `W[row * dimBOuter + col]` :
        `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
        W[row * dimBOuter + col] : 0`;

    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      if (this.hasPreluActivationWeights) {
        activationSnippet = `float activation(float a, ivec4 outCoord) {
                  float b = getPreluActivationWeightsAtOutCoords(outCoord);
                  ${this.activation}
                }`;
      } else {
        activationSnippet = `
                  float activation(float a, ivec4 outCoord) {
                    ${this.activation}
                  }
                `;
      }

      applyActivationSnippet = `value = activation(value, outCoord);`;
    }

    const addBiasSnippet =
        this.addBias ? 'value += getBiasAtOutCoords(outCoord);' : '';

    const userCode = `
    ${activationSnippet}
    ${matMulSource}

    int batch;
    int dimAOuter = ${this.outputShape[1]} * ${this.outputShape[2]};
    int dimBOuter = ${this.outputShape[3]};
    int dimInner = filterDims[0] * filterDims[1] * ${this.convInfo.inShape[3]};
    float mm_readA(int row, int col) {
      ${sampleA}
    }

    float mm_readB(int row, int col) {
      return ${sampleB};
    }

    void mm_write(int row, int col, float value) {
      ivec4 outCoord = ivec4(
          batch,
          row / ${this.outputShape[2]},
          row % ${this.outputShape[2]},
          col);
      ${addBiasSnippet}
      ${applyActivationSnippet}
      result[getFlatIndex(outCoord, ${
        getShapeCoords(this.outputShape)})] = value;
    }

    void main() {
      batch = int(gl_GlobalInvocationID.z);

      mm_matMul(dimAOuter, dimInner, dimBOuter);
    }
  `;
    return userCode;
  }
}
