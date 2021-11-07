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

import {backend_util} from '@tensorflow/tfjs-core';

import {getGlobalIndexString, getMainHeaderString} from '../shader_preprocessor';
import {computeDispatch} from '../webgpu_util';

import {mapActivationToShaderProgram} from './activation_util';
import {WebGPUProgram} from './webgpu_program';

export class DepthwiseConv2D3x3ProgramChannelFirst implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms =
      'pad : vec2<i32>; stride : vec2<i32>; dilation : vec2<i32>; inDims : vec2<i32>;';
  workGroupSize: [number, number, number] = [4, 4, 4];
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivation: boolean;
  isVec4 = true;

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: backend_util.Activation = null, hasPreluActivation = false) {
    this.outputShape = [convInfo.outShape[0], convInfo.outShape[3],
        convInfo.outShape[2], convInfo.outShape[1]];
    this.dispatchLayout = {x: [0, 1], y: [2], z: [3]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize, [1, 4, 4]);

    /*util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');*/

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

    this.shaderKey = `depthwise_channel_first_3x3_${activation}`;
  }

  getUserCode(): string {
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp =
          mapActivationToShaderProgram(this.activation, this.isVec4);
      if (this.hasPreluActivation) {
        activationSnippet =
            `fn activation(a : vec4<f32>, globalId : vec3<u32>, globalIndex : i32) -> vec4<f32> {
          let b = getPreluActivationWeightsAtOutCoordsByGlobalId(globalId, globalIndex);
          ${activationOp}
        }`;
      } else {
        activationSnippet = `
        fn activation(a : vec4<f32>, globalId : vec3<u32>, globalIndex : i32) -> vec4<f32> {
            ${activationOp}
          }
        `;
      }

      applyActivationSnippet =
          `dotProd[i] = activation(dotProd[i], globalId, index);`;
    }

    const addBiasSnippet = this.addBias ?
        'dotProd[i] = dotProd[i] + getBiasAtOutCoordsByCoords(coords);' :
        '';

    const userCode = `
      ${activationSnippet}

      ${getMainHeaderString()} {
        ${getGlobalIndexString()}
        let batch = 0;
        let d2 = i32(globalId.x);
        let r = i32(globalId.y) * 4;
        let c = i32(globalId.z) * 4;
        let xRCCorner = vec2<i32>(r, c) * uniforms.stride - uniforms.pad;
        let d1 = d2;
        let q = 0;

        let xRCorner = xRCCorner.x;
        let xCCorner = xRCCorner.y;

        var wVals : array<f32, 9>;
        wVals[0] = getW(0, 0, d1, q);
        wVals[1] = getW(0, 1, d1, q);
        wVals[2] = getW(0, 2, d1, q);
        wVals[3] = getW(1, 0, d1, q);
        wVals[4] = getW(1, 1, d1, q);
        wVals[5] = getW(1, 2, d1, q);
        wVals[6] = getW(2, 0, d1, q);
        wVals[7] = getW(2, 1, d1, q);
        wVals[8] = getW(2, 2, d1, q);

        var xVals : array<array<f32, 6>, 6>;
        for (var wR = 0; wR < 6; wR = wR + 1) {
          let xR = xRCorner + wR * uniforms.dilation[0];
          for (var wC = 0; wC < 6; wC = wC + 1) {
            let xC = xCCorner + wC * uniforms.dilation[1];
            if (xR < 0 || xR >= uniforms.inDims[0] || xC < 0 || xC >= uniforms.inDims[1]) {
              xVals[wR][wC] = f32(0.0);
            } else {
              xVals[wR][wC] = getX(batch, d1, xR, xC);
            }
          }
        }

        var dotProd : array<vec4<f32>, 4>;
        dotProd[0] = vec4<f32>(0.0);
        dotProd[1] = vec4<f32>(0.0);
        dotProd[2] = vec4<f32>(0.0);
        dotProd[3] = vec4<f32>(0.0);

        for (var idx = 0; idx < 4; idx = idx + 1) {
          for (var wR = 0; wR < 3; wR = wR + 1) {
            for (var wC = 0; wC < 3; wC = wC + 1) {
              let indexW = wR * 3 + wC;
              dotProd[idx][0] = dotProd[0][0] + xVals[idx + wR][0 + wC] * wVals[indexW];
              dotProd[idx][1] = dotProd[1][1] + xVals[idx + wR][1 + wC] * wVals[indexW];
              dotProd[idx][2] = dotProd[2][2] + xVals[idx + wR][2 + wC] * wVals[indexW];
              dotProd[idx][3] = dotProd[3][3] + xVals[idx + wR][3 + wC] * wVals[indexW];
            }
          }
        }

        for (var i = 0; i < 4; i = i + 1) {
          let coords = vec4<i32>(batch, d2, r + i, c);
          if (coordsInBounds4D(coords, uniforms.outShape)) {
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
