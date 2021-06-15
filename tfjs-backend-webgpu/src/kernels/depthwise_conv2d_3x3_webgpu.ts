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
import {computeDispatch} from '../webgpu_util';
import {mapActivationToShaderProgram} from './activation_util';
import {WebGPUProgram} from './webgpu_program';

export class DepthwiseConv2D3x3Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation, inDims;';
  workGroupSize: [number, number, number] = [4, 4, 4];
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivation: boolean;
  isVec4 = true;

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: backend_util.Activation = null, hasPreluActivation = false) {
    this.outputShape = convInfo.outShape;
    this.dispatchLayout = {x: [0, 1], y: [2], z: [3]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize, [1, 4, 4]);

    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');

    if (addBias) {
      this.variableNames.push('bias');
    }
    if (hasPreluActivation) {
      this.variableNames.push('preluActivationWeights');
    }

    this.convInfo = convInfo;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivation = hasPreluActivation;

    this.shaderKey = `depthwise3x3_${activation}`;
  }

  getUserCode(): string {
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp =
          mapActivationToShaderProgram(this.activation, this.isVec4);
      if (this.hasPreluActivation) {
        activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords(coords);
          ${activationOp}
        }`;
      } else {
        activationSnippet = `
        vec4 activation(vec4 a) {
            ${activationOp}
          }
        `;
      }

      applyActivationSnippet = `dotProd[i] = activation(dotProd[i]);`;
    }

    const addBiasSnippet =
        this.addBias ? 'dotProd[i] += getBiasAtOutCoords(coords);' : '';

    const userCode = `
      ${activationSnippet}

      void main() {
        int batch = 0;
        int r = int(gl_GlobalInvocationID.x);
        int c = int(gl_GlobalInvocationID.y) * 4;
        int d2= int(gl_GlobalInvocationID.z) * 4;
        ivec2 xRCCorner = ivec2(r, c) * stride - pad;
        int d1 = d2;
        int q = 0;

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        vec4 wVals[9];
        wVals[0] = getW(0, 0, d1, q);
        wVals[1] = getW(0, 1, d1, q);
        wVals[2] = getW(0, 2, d1, q);
        wVals[3] = getW(1, 0, d1, q);
        wVals[4] = getW(1, 1, d1, q);
        wVals[5] = getW(1, 2, d1, q);
        wVals[6] = getW(2, 0, d1, q);
        wVals[7] = getW(2, 1, d1, q);
        wVals[8] = getW(2, 2, d1, q);

        vec4 xVals[3][6];
        for (int wR = 0; wR < 3; wR++) {
          int xR = xRCorner + wR * dilation[0];
          for (int wC = 0; wC < 6; wC++) {
            int xC = xCCorner + wC * dilation[1];
            if (xR < 0 || xR >= inDims[0] || xC < 0 || xC >= inDims[1]) {
              xVals[wR][wC] = vec4(0.0, 0.0, 0.0, 0.0);
            } else {
              xVals[wR][wC] = getX(batch, xR, xC, d1);
            }
          }
        }

        vec4 dotProd[4];
        dotProd[0] = vec4(0.0, 0.0, 0.0, 0.0);
        dotProd[1] = vec4(0.0, 0.0, 0.0, 0.0);
        dotProd[2] = vec4(0.0, 0.0, 0.0, 0.0);
        dotProd[3] = vec4(0.0, 0.0, 0.0, 0.0);

        for (int wR = 0; wR < 3; wR++) {
          for (int wC = 0; wC < 3; wC++) {
            int indexW = wR * 3 + wC;
            dotProd[0] += xVals[wR][0 + wC] * wVals[indexW];
            dotProd[1] += xVals[wR][1 + wC] * wVals[indexW];
            dotProd[2] += xVals[wR][2 + wC] * wVals[indexW];
            dotProd[3] += xVals[wR][3 + wC] * wVals[indexW];
          }
        }

        for (int i = 0; i < 4; i++)
        {
          ivec4 coords = ivec4(batch, r, c + i, d2);
          if (coordsInBounds(coords, outShape)) {
            ${addBiasSnippet}
            ${applyActivationSnippet}
            setOutput(coords[0], coords[1], coords[2], coords[3], dotProd[i]);
          }
        }
      }
    `;
    return userCode;
  }
}
