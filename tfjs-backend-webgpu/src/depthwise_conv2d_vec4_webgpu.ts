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
import {activationFnSnippet, biasActivationSnippet} from './activation_util';
import {getWorkGroupSizeString, WebGPUProgram} from './webgpu_program';
import {computeDispatch} from './webgpu_util';

export class DepthwiseConv2DVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'pad : vec2<i32>, inDims : vec2<i32>,';
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
    this.dispatchLayout = {x: [3], y: [2], z: [0, 1]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize, [4, 4, 1]);

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

    this.shaderKey = `depthwiseVec4_${activation}_${
        this.convInfo.filterHeight}_${this.convInfo.filterWidth}`;
  }

  getUserCode(): string {
    // Here 4 is the work per thread in X dimension.
    const xNumber = 4 + this.convInfo.filterWidth - 1;
    const userCode = `
      ${activationFnSnippet(this.activation, this.hasPreluActivation, true, 4)}
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
        let batch = i32(globalId.z) / uniforms.outShape[1];
        let r = i32(globalId.z) % uniforms.outShape[1];
        let c = i32(globalId.y) * 4;
        let d1 = i32(globalId.x) * 4;
        let xRCCorner = vec2<i32>(r, c) - uniforms.pad;

        let xRCorner = xRCCorner.x;
        let xCCorner = xRCCorner.y;
        var xVals : array<vec4<f32>, ${xNumber}>;
        var dotProd : array<vec4<f32>, 4>;
        dotProd[0] = vec4<f32>(0.0);
        dotProd[1] = vec4<f32>(0.0);
        dotProd[2] = vec4<f32>(0.0);
        dotProd[3] = vec4<f32>(0.0);

        // Use constant instead of uniform can give better performance.
        for (var wR = 0; wR < ${this.convInfo.filterHeight}; wR = wR + 1) {
          let xR = xRCorner + wR;
          for (var i = 0; i < ${xNumber}; i++)
          {
            xVals[i] = readX(batch, xR, xCCorner + i, d1);
          }
          for (var wC = 0; wC < ${this.convInfo.filterWidth}; wC = wC + 1) {
            let wValue = getW(wR, wC, d1, 0);
            dotProd[0] = dotProd[0] + xVals[0 + wC] * wValue;
            dotProd[1] = dotProd[1] + xVals[1 + wC] * wValue;
            dotProd[2] = dotProd[2] + xVals[2 + wC] * wValue;
            dotProd[3] = dotProd[3] + xVals[3 + wC] * wValue;
          }
        }

        for (var i = 0; i < 4; i = i + 1) {
          let coords = vec4<i32>(batch, r, c + i, d1);
          if (coordsInBounds4D(coords, uniforms.outShape)) {
            var value = dotProd[i];
            ${biasActivationSnippet(this.addBias, this.activation)}
            setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
          }
        }
      }
    `;
    return userCode;
  }
}
