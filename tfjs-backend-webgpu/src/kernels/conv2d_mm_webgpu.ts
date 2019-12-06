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

  constructor(convInfo: backend_util.Conv2DInfo, workPerThread: number) {
    this.outputShape = convInfo.outShape;

    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');
    this.dispatchLayout = {x: [1, 2], y: [3], z: [0]};
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
    const tileInner = tileBOuter;
    const tileSizeA = [tileAOuter, tileInner];
    const tileSizeB = [tileInner, tileBOuter];
    const dimAOuter = this.outputShape[3];
    const dimBOuter = this.outputShape[1] * this.outputShape[2];
    const dimInner =
        convInfo.filterHeight * convInfo.filterWidth * convInfo.inChannels;
    const fitA = tilesFitEvenlyIntoShape(tileSizeA, [dimAOuter, dimInner]);
    const sampleA = fitA ?
        `W[getFlatIndex(coord, shape)]` :
        `coordsInBounds(coord, shape) ? W[getFlatIndex(coord, shape)] : 0`;
    const fitB = tilesFitEvenlyIntoShape(tileSizeB, [dimInner, dimBOuter]);
    const sampleB = fitB ?
        `x[getFlatIndex(coord, xShape)]` :
        `coordsInBounds(coord, xShape) ? x[getFlatIndex(coord, xShape)] : 0`;

    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize,
        elementsPerThread);

    this.userCode = `
        ${matMulSource}

        int batch;

        float mm_readA(int row, int col) {
          int r = int(row), c = int(col);
          ivec4 coord = ivec4(
              (c / filterDims[1]) % filterDims[0],
              c % filterDims[1],
              c / (filterDims[0] * filterDims[1]),
              r);

          ivec4 shape = ivec4(filterDims, xShape[3], outShape[3]);
          return ${sampleA};
        }

        float mm_readB(int row, int col) {
          int r = int(row), c = int(col);
          int outRow = c / outShape[2];
          int outCol = c % outShape[2];

          int WRow = (r / filterDims[1]) % filterDims[0];
          int WCol = r % filterDims[1];

          ivec4 coord = ivec4(
              batch,
              pad[0] + outRow * stride[0] + dilation[0] * WRow,
              pad[1] + outCol * stride[1] + dilation[1] * WCol,
              r / (filterDims[0] * filterDims[1]));
          return ${sampleB};
        }

        void mm_write(int row, int col, float value) {
          ivec4 outCoord = ivec4(
              batch,
              col / outShape[2],
              col % outShape[2],
              row);
          if (coordsInBounds(outCoord, outShape)) {
            result[getFlatIndex(outCoord, outShape)] = value;
          }
        }

        void main() {
          batch = int(gl_GlobalInvocationID.z);

          int dimAOuter = outShape[3];
          int dimBOuter = outShape[1] * outShape[2];
          int dimInner = filterDims[0] * filterDims[1] * xShape[3];
          mm_matMul(dimAOuter, dimInner, dimBOuter);
        }
      `;
    this.shaderKey = `conv2dmm'${elementsPerThread.join('')}${fitA}${fitB}`;
  }
}
