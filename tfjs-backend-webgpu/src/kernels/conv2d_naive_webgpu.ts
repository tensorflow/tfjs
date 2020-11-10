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

import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class Conv2DNaiveProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation;';
  workGroupSize: [number, number, number] = [4, 8, 4];

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: string = null, hasPreluActivationWeights = false) {
    this.outputShape = convInfo.outShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0, 3]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');
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
      float readInp(int batch, int row, int col, int chan) {
        ivec4 coord = ivec4(batch, row, col, chan);
        return coordsInBounds(coord, xShape) ?
          getX(batch, row, col, chan) : 0;
      }

      float readFilt(int row, int col, int xChannel, int outChannel) {
        ivec4 coord = ivec4(row, col, xChannel, outChannel);
        return coordsInBounds(coord, wShape) ?
          getW(row, col, xChannel, outChannel) : 0;
      }

      void writeResult(int batch, int row, int col, int chan, float value) {
        ivec4 coord = ivec4(batch, row, col, chan);
        if (coordsInBounds(coord, outShape)) {
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutput(batch, row, col, chan, value);
        }
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int outChannel = coords[3];

        float acc = 0.0;

        for (int row = 0; row < filterDims[0]; ++row) {
          for (int col = 0; col < filterDims[1]; ++col) {
            for (int xChannel = 0; xChannel < xShape[3]; ++xChannel) {
              float v = readInp(batch,
                  pad[0] + coords[1] * stride[0] + dilation[0] * row,
                  pad[1] + coords[2] * stride[1] + dilation[1] * col,
                  xChannel);
              float f = readFilt(row, col, xChannel, outChannel);
              acc += v * f;
            }
          }
        }

        writeResult(batch, coords[1], coords[2], outChannel, acc);
      }
    `;
    this.shaderKey = 'conv2dnaive';
  }
}
