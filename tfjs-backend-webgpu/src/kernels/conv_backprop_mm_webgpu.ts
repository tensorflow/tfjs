/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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

import {computeDispatch, computeWorkGroupSizeForConv2d, computeWorkPerThreadForConv2d} from '../webgpu_util';

import {makeMatMulPackedSource} from './matmul_packed_webgpu';
import {WebGPUProgram} from './webgpu_program';

export class Conv2DDerInputMMProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pads, stride; ivec4 outBackprop;';
  workGroupSize: [number, number, number];
  elementsPerThread: [number, number, number];

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.inShape;

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

    this.shaderKey = `conv2DDerInputMM_${this.elementsPerThread}`;
  }

  getUserCode(): string {
    const matMulSource = makeMatMulPackedSource(this.elementsPerThread);

    const readASnippet = `
    int outRow = row / outShape[2];
    int outCol = row % outShape[2];

    int WRow = col / (filterDims[1] * outBackprop[3]);
    int WCol = (col / outBackprop[3]) % filterDims[1];
    float xR = float(outRow - pads[0] + WRow) / float(stride[0]);
    float xC = float(outCol - pads[1] + WCol) / float(stride[1]);
    if (xR < 0.0 || xR >= float(outBackprop[1]) || fract(xR) > 0.0) {
      return 0;
    }
    if (xC < 0.0 || xC >= float(outBackprop[2]) || fract(xC) > 0.0) {
      return 0;
    }
    ivec4 coord = ivec4(
        batch,
        int(xR),
        int(xC),
        col % outBackprop[3]);
    return x[getFlatIndex(coord, xShape)];`;

    const sampleA = `if (row < dimAOuter && col < dimInner) {
      ${readASnippet}
    } else {
      return 0;
    }`;

    const userCode = `
    ${matMulSource}

    int batch;
    int dimAOuter = outShape[1] * outShape[2];
    int dimBOuter = outShape[3];
    int dimInner = filterDims[0] * filterDims[1] * outBackprop[3];

    float mm_readA(int row, int col) {
      ${sampleA}
    }

    float mm_readB(int row, int col) {
      if (row < dimInner && col < dimBOuter)
      {
        int WRow = row / (filterDims[1] * outBackprop[3]);
        int WCol = (row / outBackprop[3]) % filterDims[1];
        ivec4 coord = ivec4(
            filterDims.x - 1 - WRow,
            filterDims.y - 1 - WCol,
            col,
            row % outBackprop[3]);
        return W[getFlatIndex(coord, wShape)];
      } else
      {
        return 0;
      }
    }

    void mm_write(int row, int col, float value) {
      ivec4 outCoord = ivec4(
          batch,
          row / outShape[2],
          row % outShape[2],
          col);
      result[getFlatIndex(outCoord, outShape)] = value;
    }

    void main() {
      batch = int(gl_GlobalInvocationID.z);

      mm_matMul(dimAOuter, dimInner, dimBOuter);
    }
  `;
    return userCode;
  }
}
