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
import {mapActivationToShaderProgram} from './activation_util';
import {getWorkGroupSizeString, WebGPUProgram} from './webgpu_program';
import {computeDispatch} from './webgpu_util';

export class DepthwiseConv2D3x3Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms =
      'pad : vec2<i32>, stride : vec2<i32>, dilation : vec2<i32>, inDims : vec2<i32>,';
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
        activationSnippet =
            `fn activation(a : vec4<f32>, outCoord : vec4<i32>) -> vec4<f32> {
          let b = getPreluActivationWeightsByOutputCoords(outCoord);
          ${activationOp}
        }`;
      } else {
        activationSnippet = `
        fn activation(a : vec4<f32>, outCoord : vec4<i32>) -> vec4<f32> {
            ${activationOp}
          }
        `;
      }

      applyActivationSnippet = `dotProd[i] = activation(dotProd[i], coords);`;
    }

    const addBiasSnippet = this.addBias ?
        'dotProd[i] = dotProd[i] + getBiasByOutputCoords(coords);' :
        '';

    const userCode = `
      ${activationSnippet}
      fn readX(batch : i32, row : i32, col : i32, channel : i32) -> vec4<f32> {
        var value = vec4<f32>(0.0);
        if (row >=0 && row < uniforms.inDims[0] && col >=0 && col < uniforms.inDims[1])
        {
          value = getX(batch, row, col, channel);
        }
        return value;
      }
      ${getWorkGroupSizeString()}
      fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
        let batch = 0;
        let r = i32(globalId.x);
        let c = i32(globalId.y) * 4;
        let d2 = i32(globalId.z) * 4;
        let xRCCorner = vec2<i32>(r, c) * uniforms.stride - uniforms.pad;
        let d1 = d2;
        let q = 0;

        let xRCorner = xRCCorner.x;
        let xCCorner = xRCCorner.y;
        var xVals : array<vec4<f32>, 6>;
        var dotProd : array<vec4<f32>, 4>;
        dotProd[0] = vec4<f32>(0.0);
        dotProd[1] = vec4<f32>(0.0);
        dotProd[2] = vec4<f32>(0.0);
        dotProd[3] = vec4<f32>(0.0);

        for (var wR = 0; wR < 3; wR = wR + 1) {
          let xR = xRCorner + wR;
          xVals[0] = readX(batch, xR, xCCorner, d1);
          xVals[1] = readX(batch, xR, xCCorner + 1, d1);
          xVals[2] = readX(batch, xR, xCCorner + 2, d1);
          xVals[3] = readX(batch, xR, xCCorner + 3, d1);
          xVals[4] = readX(batch, xR, xCCorner + 4, d1);
          xVals[5] = readX(batch, xR, xCCorner + 5, d1);
          for (var wC = 0; wC < 3; wC = wC + 1) {
            let wValue = getW(wR, wC, d1, q);
            dotProd[0] = dotProd[0] + xVals[0 + wC] * wValue;
            dotProd[1] = dotProd[1] + xVals[1 + wC] * wValue;
            dotProd[2] = dotProd[2] + xVals[2 + wC] * wValue;
            dotProd[3] = dotProd[3] + xVals[3 + wC] * wValue;
          }
        }

        for (var i = 0; i < 4; i = i + 1) {
          let coords = vec4<i32>(batch, r, c + i, d2);
          if (coordsInBounds4D(coords, uniforms.outShape)) {
            ${addBiasSnippet}
            ${applyActivationSnippet}
            setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], dotProd[i]);
          }
        }
      }
    `;
    return userCode;
  }
}
