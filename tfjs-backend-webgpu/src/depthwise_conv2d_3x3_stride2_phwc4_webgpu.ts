/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import {getMainHeaderString} from './shader_preprocessor';
import {WebGPUProgram} from './webgpu_program';
import {computeDispatch} from './webgpu_util';

export class DepthwiseConv2D3x3Phwc4Stride2Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['X', 'W'];
  uniforms =
      'pad : vec2<i32>; stride : vec2<i32>; sliceStride: i32; dstSlice: i32;';
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
    this.dispatchLayout = {x: [2], y: [1], z: [0, 3]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize, [1, 2, 4]);

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

    this.shaderKey = `depthwise3x3phwc4Stride2_${activation}`;
  }

  getUserCode(): string {
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp =
          mapActivationToShaderProgram(this.activation, this.isVec4);
      activationSnippet = `
          fn activation(a : vec4<f32>) -> vec4<f32> {
            ${activationOp}
          }`;
      applyActivationSnippet = `value = activation(valueIn);`;
    }
    const userCode = `
      ${activationSnippet}
      fn writeResult(col : i32, row : i32, chan : i32, valueIn : vec4<f32>) {
        let dstIndex = (chan * uniforms.outShape[1] + row) * uniforms.outShape[2] + col;
        var value = valueIn;
        ${applyActivationSnippet}
        result.numbers[dstIndex] = value;
      }

      ${getMainHeaderString()}
      let DST_X = i32(globalId.x);
      let DST_Y = i32(globalId.y) * 2;
      let DST_S = i32(globalId.z);
      if (DST_X >= uniforms.outShape[2] || DST_Y >= uniforms.outShape[1] || DST_S >= uniforms.dstSlice) {
        return;
      }
      var r0 = vec4<f32>(0.0);
      var l0 = vec4<f32>(0.0);

      var s0 = vec4<f32>(0.0);
      var s1 = vec4<f32>(0.0);
      var s2 = vec4<f32>(0.0);

      var x0 = DST_X * uniforms.stride[1] - uniforms.pad[1];
      var x1 = x0 + 1;
      var x2 = x0 + 2;
      var y0 = DST_Y * uniforms.stride[0] - uniforms.pad[0];
      var y1 = y0 + 1;
      var y2 = y0 + 2;
      var y3 = y0 + 3;
      var y4 = y0 + 4;

      var fLoc = DST_S * 9;
      let x0_in = x0 >= 0 && x0 < uniforms.xShape[2];
      let x1_in = x1 >= 0 && x1 < uniforms.xShape[2];
      let x2_in = x2 >= 0 && x2 < uniforms.xShape[2];

      let y0_in = y0 >= 0 && y0 < uniforms.xShape[1];
      let y1_in = y1 >= 0 && y1 < uniforms.xShape[1];
      let y2_in = y2 >= 0 && y2 < uniforms.xShape[1];
      let y3_in = y3 >= 0 && y3 < uniforms.xShape[1];
      let y4_in = y4 >= 0 && y4 < uniforms.xShape[1];

      x0 = clamp(x0, 0, uniforms.xShape[2] - 1);
      x1 = clamp(x1, 0, uniforms.xShape[2] - 1);
      x2 = clamp(x2, 0, uniforms.xShape[2] - 1);
      y0 = clamp(y0, 0, uniforms.xShape[1] - 1);
      y1 = clamp(y1, 0, uniforms.xShape[1] - 1);
      y2 = clamp(y2, 0, uniforms.xShape[1] - 1);
      y3 = clamp(y3, 0, uniforms.xShape[1] - 1);
      y4 = clamp(y4, 0, uniforms.xShape[1] - 1);

      let srcLoc = DST_S * uniforms.sliceStride;
      s0 = X.numbers[srcLoc + y0 * uniforms.xShape[2] + x0] * f32(x0_in && y0_in);
      s1 = X.numbers[srcLoc + y0 * uniforms.xShape[2] + x1] * f32(x1_in && y0_in);
      s2 = X.numbers[srcLoc + y0 * uniforms.xShape[2] + x2] * f32(x2_in && y0_in);

      r0 = r0 + (W.numbers[fLoc + 0] * s0);
      r0 = r0 + (W.numbers[fLoc + 1] * s1);
      r0 = r0 + (W.numbers[fLoc + 2] * s2);

      s0 = X.numbers[srcLoc + y1 * uniforms.xShape[2] + x0] * f32(x0_in && y1_in);
      s1 = X.numbers[srcLoc + y1 * uniforms.xShape[2] + x1] * f32(x1_in && y1_in);
      s2 = X.numbers[srcLoc + y1 * uniforms.xShape[2] + x2] * f32(x2_in && y1_in);
      r0 = r0 + (W.numbers[fLoc + 3] * s0);
      r0 = r0 + (W.numbers[fLoc + 4] * s1);
      r0 = r0 + (W.numbers[fLoc + 5] * s2);

      s0 = X.numbers[srcLoc + y2 * uniforms.xShape[2] + x0] * f32(x0_in && y2_in);
      s1 = X.numbers[srcLoc + y2 * uniforms.xShape[2] + x1] * f32(x1_in && y2_in);
      s2 = X.numbers[srcLoc + y2 * uniforms.xShape[2] + x2] * f32(x2_in && y2_in);
      r0 = r0 + (W.numbers[fLoc + 6] * s0);
      r0 = r0 + (W.numbers[fLoc + 7] * s1);
      r0 = r0 + (W.numbers[fLoc + 8] * s2);
      l0 = l0 + (W.numbers[fLoc + 0] * s0);
      l0 = l0 + (W.numbers[fLoc + 1] * s1);
      l0 = l0 + (W.numbers[fLoc + 2] * s2);

      s0 = X.numbers[srcLoc + y3 * uniforms.xShape[2] + x0] * f32(x0_in && y3_in);
      s1 = X.numbers[srcLoc + y3 * uniforms.xShape[2] + x1] * f32(x1_in && y3_in);
      s2 = X.numbers[srcLoc + y3 * uniforms.xShape[2] + x2] * f32(x2_in && y3_in);
      l0 = l0 + (W.numbers[fLoc + 3] * s0);
      l0 = l0 + (W.numbers[fLoc + 4] * s1);
      l0 = l0 + (W.numbers[fLoc + 5] * s2);

      s0 = X.numbers[srcLoc + y4 * uniforms.xShape[2] + x0] * f32(x0_in && y4_in);
      s1 = X.numbers[srcLoc + y4 * uniforms.xShape[2] + x1] * f32(x1_in && y4_in);
      s2 = X.numbers[srcLoc + y4 * uniforms.xShape[2] + x2] * f32(x2_in && y4_in);
    l0 = l0 + (W.numbers[fLoc + 6] * s0);
    l0 = l0 + (W.numbers[fLoc + 7] * s1);
    l0 = l0 + (W.numbers[fLoc + 8] * s2);

      {
       let bias_val = ${this.addBias ? 'bias.numbers[DST_S]' : '0.0'};
      {
        let res = r0 + bias_val;
        writeResult(DST_X, DST_Y, DST_S, res);
      }
      if (DST_Y + 1 < uniforms.outShape[1]) {
        let res = l0 + bias_val;
        writeResult(DST_X, DST_Y + 1, DST_S, res);
      }
      }

      }
    `;
    return userCode;
  }
}
