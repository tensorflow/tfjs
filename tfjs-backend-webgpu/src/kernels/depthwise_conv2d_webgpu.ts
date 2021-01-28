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
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
import {WebGPUProgram} from './webgpu_program';

export class DepthwiseConv2DProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation, inDims;';
  // This is an experimental value.
  workGroupSize: [number, number, number] = [256, 1, 1];
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: string;
  hasPreluActivation: boolean;

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: string = null, hasPreluActivation = false) {
    this.outputShape = convInfo.outShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

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

    this.shaderKey = `depthwise_${activation}`;
  }

  getUserCode(): string {
    const channelMul = this.convInfo.outChannels / this.convInfo.inChannels;
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      if (this.hasPreluActivation) {
        activationSnippet = `float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${this.activation}
        }`;
      } else {
        activationSnippet = `
          float activation(float a) {
            ${this.activation}
          }
        `;
      }

      applyActivationSnippet = `dotProd = activation(dotProd);`;
    }

    const addBiasSnippet =
        this.addBias ? 'dotProd += getBiasAtOutCoords();' : '';

    const userCode = `
      ${activationSnippet}

      void writeResult(int batch, int row, int col, int chan, float value) {
        ivec4 coord = ivec4(batch, row, col, chan);
        if (coordsInBounds(coord, ${getShapeCoords(this.outputShape)})) {
          setOutput(batch, row, col, chan, value);
        }
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        ivec2 xRCCorner = coords.yz * stride - pad;
        int d2 = coords[3];
        int d1 = d2 / ${channelMul};
        int q = d2 - d1 * ${channelMul};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        // TODO(xing.xu): Flatten the two for loops and vec4 the operations.
        for (int wR = 0; wR < filterDims[0]; wR++) {
          int xR = xRCorner + wR * dilation[0];

          if (xR < 0 || xR >= inDims[0]) {
            continue;
          }

          for (int wC = 0; wC < filterDims[1]; wC++) {
            int xC = xCCorner + wC * dilation[1];

            if (xC < 0 || xC >= inDims[1]) {
              continue;
            }

            float xVal = getX(batch, xR, xC, d1);
            float wVal = getW(wR, wC, d1, q);
            dotProd += xVal * wVal;
          }
        }

        ${addBiasSnippet}
        ${applyActivationSnippet}
        writeResult(batch, coords[1], coords[2], d2, dotProd);
      }
    `;
    return userCode;
  }
}
