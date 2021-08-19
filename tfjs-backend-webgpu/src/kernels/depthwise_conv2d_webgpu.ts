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

import {getWorkGroupSizeStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {mapActivationToShaderProgram} from './activation_util';
import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class DepthwiseConv2DProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 pad, stride, dilation, inDims;';
  uniformsWgsl =
      `pad : vec2<u32>; stride : vec2<u32>; dilation : vec2<u32>; inDims : vec2<u32>;`;
  // This is an experimental value.
  workGroupSize: [number, number, number] = [256, 1, 1];
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivation: boolean;
  useWgsl: boolean;

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
    this.useWgsl = getUseWgsl();

    this.shaderKey = `depthwise_${this.convInfo.filterHeight}_${
        this.convInfo.filterWidth}_${this.activation}_${
        this.convInfo.outChannels / this.convInfo.inChannels}`;
  }

  getUserCode(): string {
    const channelMul = this.convInfo.outChannels / this.convInfo.inChannels;
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(this.activation);
      if (this.hasPreluActivation) {
        activationSnippet = `float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${activationOp}
        }`;
      } else {
        activationSnippet = `
          float activation(float a) {
            ${activationOp}
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
        if (coordsInBounds(coord, outShape)) {
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

        int inputRowStart = xRCCorner.x;
        int inputColStart = xRCCorner.y;
        int inputRowEnd = inputRowStart + ${
        this.convInfo.filterHeight} * dilation[0];
        int inputColEnd = inputColStart + ${
        this.convInfo.filterWidth} * dilation[1];

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        // Extract if checking out of for loop for performance.
        if (inputRowStart >= 0 && inputColStart >= 0 &&
          inputRowEnd < inDims[0] && inputColEnd < inDims[1])
          {
            // Here using a constant value |this.convInfo.filterHeight| instead
            // of uniform value is in order to loop unrolling.
            for (int wR = 0; wR < ${this.convInfo.filterHeight}; wR++) {
              int xR = inputRowStart + wR * dilation[0];

              for (int wC = 0; wC < ${this.convInfo.filterWidth}; wC++) {
                int xC = inputColStart + wC * dilation[1];

                float xVal = getX(batch, xR, xC, d1);
                float wVal = getW(wR, wC, d1, q);
                dotProd += xVal * wVal;
              }
            }
          } else {
            for (int wR = 0; wR < ${this.convInfo.filterHeight}; wR++) {
              int xR = inputRowStart + wR * dilation[0];

              if (xR < 0 || xR >= inDims[0]) {
                continue;
              }

              for (int wC = 0; wC < ${this.convInfo.filterWidth}; wC++) {
                int xC = inputColStart + wC * dilation[1];

                if (xC < 0 || xC >= inDims[1]) {
                  continue;
                }

                float xVal = getX(batch, xR, xC, d1);
                float wVal = getW(wR, wC, d1, q);
                dotProd += xVal * wVal;
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

  getUserCodeWgsl(): string {
    const channelMul = this.convInfo.outChannels / this.convInfo.inChannels;
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp =
          mapActivationToShaderProgram(this.activation, false, this.useWgsl);
      if (this.hasPreluActivation) {
        activationSnippet =
            `fn activation(a : f32, globalId : vec3<u32>) -> f32 {
          let b = getPreluActivationWeightsAtOutCoordsByGlobalId(globalId);
          ${activationOp}
        }`;
      } else {
        activationSnippet = `
          fn activation(a : f32, globalId : vec3<u32>) -> f32 {
            ${activationOp}
          }
        `;
      }

      applyActivationSnippet = `dotProd = activation(dotProd, globalId);`;
    }

    const addBiasSnippet = this.addBias ?
        'dotProd = dotProd + getBiasAtOutCoordsByGlobalId(globalId);' :
        '';

    const userCode = `
      ${activationSnippet}

      fn writeResult(batch : u32, row : u32, col : u32, chan : u32, value : f32) {
        let coord = vec4<u32>(batch, row, col, chan);
        if (coordsInBounds4D(coord, uniforms.outShape)) {
          setOutput(batch, row, col, chan, value);
        }
      }

      ${getWorkGroupSizeStringWgsl(this.workGroupSize)}
      fn main([[builtin(global_invocation_id)]] globalId : vec3<u32>) {
        let coords = getOutputCoords(globalId);
        let batch = coords[0];
        let xRCCorner = vec2<i32>(coords.yz * uniforms.stride - uniforms.pad);
        let d2 = coords[3];
        let d1 = d2 / ${channelMul}u;
        let q = d2 - d1 * ${channelMul}u;

        let inputRowStart = xRCCorner.x;
        let inputColStart = xRCCorner.y;
        let inputRowEnd = inputRowStart + i32(${
        this.convInfo.filterHeight}u * uniforms.dilation[0]);
        let inputColEnd = inputColStart + i32(${
        this.convInfo.filterWidth}u * uniforms.dilation[1]);

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        var dotProd = 0.0;

        // Extract if checking out of for loop for performance.
        if (inputRowStart >= 0 && inputColStart >= 0 &&
          inputRowEnd < i32(uniforms.inDims[0]) && inputColEnd < i32(uniforms.inDims[1]))
          {
            // Here using a constant value |this.convInfo.filterHeight| instead
            // of uniform value is in order to loop unrolling.
            for (var wR = 0u; wR < ${
        this.convInfo.filterHeight}u; wR = wR + 1u) {
              let xR = inputRowStart + i32(wR * uniforms.dilation[0]);

              for (var wC = 0u; wC < ${
        this.convInfo.filterWidth}u; wC = wC + 1u) {
                let xC = inputColStart + i32(wC * uniforms.dilation[1]);

                let xVal = getX(batch, u32(xR), u32(xC), d1);
                let wVal = getW(wR, u32(wC), d1, q);
                dotProd = dotProd + xVal * wVal;
              }
            }
          } else {
            for (var wR = 0u; wR < ${
        this.convInfo.filterHeight}u; wR = wR + 1u) {
              let xR = inputRowStart + i32(wR * uniforms.dilation[0]);

              if (xR < 0 || xR >= i32(uniforms.inDims[0])) {
                continue;
              }

              for (var wC = 0u; wC < ${
        this.convInfo.filterWidth}u; wC = wC + 1u) {
                let xC = inputColStart + i32(wC * uniforms.dilation[1]);

                if (xC < 0 || xC >= i32(uniforms.inDims[1])) {
                  continue;
                }

                let xVal = getX(batch, u32(xR), u32(xC), d1);
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
