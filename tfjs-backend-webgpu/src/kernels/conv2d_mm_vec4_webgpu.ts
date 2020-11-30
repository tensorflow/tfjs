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

import {backend_util, util} from '@tensorflow/tfjs-core';

import {getShapeCoords} from '../shader_preprocessor';
import {computeDispatch, tilesFitEvenlyIntoShape} from '../webgpu_util';

import {makeMatMulPackedVec4Source} from './matmul_packed_vec4_webgpu';
import {WebGPUProgram} from './webgpu_program';

export class Conv2DMMVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation;';
  workGroupSize: [number, number, number];
  isVec4 = true;

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: string = null, hasPreluActivationWeights = false) {
    this.outputShape = convInfo.outShape;

    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');
    this.dispatchLayout = {x: [3], y: [1, 2], z: [0]};
    this.workGroupSize = [16, 16, 1];
    const elementsPerThread: [number, number, number] = [4, 4, 1];
    const matMulSource = makeMatMulPackedVec4Source(elementsPerThread);

    const tileAOuter = this.workGroupSize[1] * elementsPerThread[1];
    const tileBOuter = this.workGroupSize[0] * elementsPerThread[0];
    const tileInner = tileBOuter;

    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];
    const dimAOuter = this.outputShape[1] * this.outputShape[2];
    const dimBOuter = this.outputShape[3];
    const dimInner =
        convInfo.filterHeight * convInfo.filterWidth * convInfo.inChannels;
    const fitA = tilesFitEvenlyIntoShape(tileSizeA, [dimAOuter, dimInner]);
    const sampleA = fitA ?
        `x[getFlatIndex(coord, ${getShapeCoords(convInfo.inShape)}) / 4]` :
        `coordsInBounds(coord, ${
            getShapeCoords(convInfo.inShape)}) ? x[getFlatIndex(coord, ${
            getShapeCoords(convInfo.inShape)}) / 4] : vec4(0.0, 0.0, 0.0, 0.0)`;
    const fitB = tilesFitEvenlyIntoShape(tileSizeB, [dimInner, dimBOuter]);
    const sampleB = fitB ?
        `W[row * dimBOuter + col]` :
        `coordsInBounds(ivec2(row, col), ivec2(dimInner * 4, dimBOuter)) ?
        W[row * dimBOuter + col] : vec4(0.0, 0.0, 0.0, 0.0)`;

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        elementsPerThread);
    const batchSize =
        this.outputShape[1] * this.outputShape[2] * this.outputShape[3] / 4;

    // TODO(jiajia.qin@intel.com): Add the fused conv2d vec4 support.
    this.userCode = `
        ${matMulSource}

        int batch;
        int dimAOuter = ${this.outputShape[1]} * ${this.outputShape[2]};
        int dimBOuter = ${this.outputShape[3] / 4};
        int dimInner = filterDims[0] * filterDims[1] * ${
        convInfo.inShape[3] / 4};
        vec4 mm_readA(int row, int col) {
          int r = int(row), c = int(col * 4);
          int outRow = r / ${this.outputShape[2]};
          int outCol = r % ${this.outputShape[2]};

          int WRow = c / (filterDims[1] * ${convInfo.inShape[3]});
          int WCol = (c / ${convInfo.inShape[3]}) % filterDims[1];

          ivec4 coord = ivec4(
              batch,
              outRow * stride[0] + dilation[0] * WRow - pad[0],
              outCol * stride[1] + dilation[1] * WCol - pad[1],
              c % ${convInfo.inShape[3]});
          return ${sampleA};
        }

        vec4 mm_readB(int row, int col) {
          return ${sampleB};
        }

        void mm_write(int row, int col, vec4 value) {
          if (row < dimAOuter && col < dimBOuter)
          {
            result[batch * ${batchSize} + row * dimBOuter + col] = value;
          }
        }

        void main() {
          batch = int(gl_GlobalInvocationID.z);

          mm_matMul(dimAOuter, dimInner, dimBOuter);
        }
      `;
  }
}
