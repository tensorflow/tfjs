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

import {computeDispatch, computeWorkGroupSizeForConv2d, computeWorkPerThreadForConv2d, tilesFitEvenlyIntoShape} from '../webgpu_util';

import {makeMatMulPackedSource} from './matmul_packed_webgpu';
import {makeMatMulSource} from './matmul_webgpu';
import {WebGPUProgram} from './webgpu_program';

export class Conv2DMMProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation;';
  workGroupSize: [number, number, number];

  constructor(
      convInfo: backend_util.Conv2DInfo, workPerThread: number, addBias = false,
      activation: string = null, hasPreluActivationWeights = false) {
    this.outputShape = convInfo.outShape;

    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');
    this.dispatchLayout = {x: [3], y: [1, 2], z: [0]};
    this.workGroupSize =
        computeWorkGroupSizeForConv2d(this.dispatchLayout, this.outputShape);
    let elementsPerThread: [number, number, number];
    let matMulSource: string;
    if (workPerThread === 0) {
      elementsPerThread = [1, 1, 1];
      matMulSource = makeMatMulSource();
    } else {
      elementsPerThread =
          computeWorkPerThreadForConv2d(this.dispatchLayout, this.outputShape);
      matMulSource = makeMatMulPackedSource(elementsPerThread);
    }

    const tileAOuter = this.workGroupSize[1] * elementsPerThread[1];
    const tileBOuter = this.workGroupSize[0] * elementsPerThread[0];
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
    const dimInner =
        convInfo.filterHeight * convInfo.filterWidth * convInfo.inChannels;
    const fitA = tilesFitEvenlyIntoShape(tileSizeA, [dimAOuter, dimInner]);
    const sampleA = fitA ?
        `x[getFlatIndex(coord, xShape)]` :
        `coordsInBounds(coord, xShape) ? x[getFlatIndex(coord, xShape)] : 0`;
    const fitB = tilesFitEvenlyIntoShape(tileSizeB, [dimInner, dimBOuter]);
    const sampleB = fitB ?
        `W[row * dimBOuter + col]` :
        `coordsInBounds(ivec2(row, col), ivec2(dimInner, dimBOuter)) ?
        W[row * dimBOuter + col] : 0`;

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        elementsPerThread);

    let activationSnippet = '', applyActivationSnippet = '';
    if (activation) {
      if (hasPreluActivationWeights) {
        activationSnippet = `float activation(float a) {
              float b = getPreluActivationWeightsAtOutCoords();
              ${activation}
            }`;
      } else {
        activationSnippet = `
              float activation(float a) {
                ${activation}
              }
            `;
      }

      applyActivationSnippet = `value = activation(value);`;
    }

    const addBiasSnippet = addBias ? 'value += getBiasAtOutCoords();' : '';
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.userCode = `
        ${activationSnippet}
        ${matMulSource}

        int batch;
        int dimAOuter = outShape[1] * outShape[2];
        int dimBOuter = outShape[3];
        int dimInner = filterDims[0] * filterDims[1] * xShape[3];
        float mm_readA(int row, int col) {
          int r = int(row), c = int(col);
          int outRow = r / outShape[2];
          int outCol = r % outShape[2];

          int WRow = c / (filterDims[1] * xShape[3]);
          int WCol = (c / xShape[3]) % filterDims[1];

          ivec4 coord = ivec4(
              batch,
              outRow * stride[0] + dilation[0] * WRow - pad[0],
              outCol * stride[1] + dilation[1] * WCol - pad[1],
              c % xShape[3]);
          return ${sampleA};
        }

        float mm_readB(int row, int col) {
          return ${sampleB};
        }

        void mm_write(int row, int col, float value) {
          ivec4 outCoord = ivec4(
              batch,
              row / outShape[2],
              row % outShape[2],
              col);
          ${addBiasSnippet}
          ${applyActivationSnippet}
          result[getFlatIndex(outCoord, outShape)] = value;
        }

        void main() {
          batch = int(gl_GlobalInvocationID.z);

          mm_matMul(dimAOuter, dimInner, dimBOuter);
        }
      `;
    this.shaderKey = `conv2dmm'${elementsPerThread.join('')}${fitA}${fitB}${
        addBiasSnippet}${activationSnippet}`;
  }
}
