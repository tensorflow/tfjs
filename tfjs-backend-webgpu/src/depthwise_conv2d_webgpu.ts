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
import {getMainHeaderString, WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class DepthwiseConv2DProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = `pad : vec2<i32>, stride : vec2<i32>, dilation : vec2<i32>,
      inDims : vec2<i32>, filterHeight : i32, filterWidth : i32,
      channelMul : i32,`;
  // This is an experimental value.
  workGroupSize: [number, number, number] = [256, 1, 1];
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivation: boolean;

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: backend_util.Activation = null, hasPreluActivation = false) {
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
    this.shaderKey = `depthwise_${this.activation}`;
  }

  getUserCode(): string {
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(this.activation, false);
      if (this.hasPreluActivation) {
        activationSnippet =
            `fn activation(a : f32, outCoord : vec4<i32>) -> f32 {
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

      applyActivationSnippet = `dotProd = activation(dotProd, coords);`;
    }

    const addBiasSnippet = this.addBias ?
        'dotProd = dotProd + getBiasByOutputCoords(coords);' :
        '';

    const userCode = `
      ${activationSnippet}

      fn writeResult(batch : i32, row : i32, col : i32, chan : i32,
          value : f32) {
        let coord = vec4<i32>(batch, row, col, chan);
        if (coordsInBounds4D(coord, uniforms.outShape)) {
          setOutputAtCoords(batch, row, col, chan, value);
        }
      }

      ${getMainHeaderString()}
        let coords = getOutputCoords();
        let batch = coords[0];
        let xRCCorner = vec2<i32>(coords.yz) * uniforms.stride - uniforms.pad;
        let d2 = coords[3];
        let d1 = d2 / uniforms.channelMul;
        let q = d2 - d1 * uniforms.channelMul;

        let inputRowStart = xRCCorner.x;
        let inputColStart = xRCCorner.y;
        let inputRowEnd = inputRowStart + uniforms.filterHeight *
            uniforms.dilation[0];
        let inputColEnd = inputColStart + uniforms.filterWidth *
            uniforms.dilation[1];

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;

        // Extract if checking out of for loop for performance.
        if (inputRowStart >= 0 && inputColStart >= 0 &&
          inputRowEnd < uniforms.inDims[0] &&
              inputColEnd < uniforms.inDims[1]) {
            // Here using a constant value |this.convInfo.filterHeight| instead
            // of uniform value is in order to loop unrolling.
            for (var wR = 0; wR < uniforms.filterHeight; wR = wR + 1) {
              let xR = inputRowStart + wR * uniforms.dilation[0];

              for (var wC = 0; wC < uniforms.filterWidth; wC = wC + 1) {
                let xC = inputColStart + wC * uniforms.dilation[1];

                let xVal = getX(batch, xR, xC, d1);
                let wVal = getW(wR, wC, d1, q);
                dotProd = dotProd + xVal * wVal;
              }
            }
          } else {
            for (var wR = 0; wR < uniforms.filterHeight; wR = wR + 1) {
              let xR = inputRowStart + wR * uniforms.dilation[0];

              if (xR < 0 || xR >= uniforms.inDims[0]) {
                continue;
              }

              for (var wC = 0; wC < uniforms.filterWidth; wC = wC + 1) {
                let xC = inputColStart + wC * uniforms.dilation[1];

                if (xC < 0 || xC >= uniforms.inDims[1]) {
                  continue;
                }

                let xVal = getX(batch, xR, xC, d1);
                let wVal = getW(wR, wC, d1, q);
                dotProd = dotProd + xVal * wVal;
              }
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
