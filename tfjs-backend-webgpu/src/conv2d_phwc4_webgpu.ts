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

export class Conv2DTFLiteProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['X', 'W'];
  uniforms =
      `filterDims : vec2<i32>; pad : vec2<i32>; stride : vec2<i32>; dilation : vec2<i32>; srcSlice: i32; sliceStride: i32; dstSlice: i32;`;
  workGroupSize: [number, number, number] = [8, 4, 1];
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  isVec4 = true;

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: backend_util.Activation = null,
      hasPreluActivationWeights = false) {
    this.outputShape = convInfo.outShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0, 3]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize, [2, 2, 2]);

    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.convInfo = convInfo;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;

    this.shaderKey = `conv2DPHWC4_${this.addBias}_${this.activation}`;
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
      let DST_X = i32(globalId.x) * 2;
      let DST_Y = i32(globalId.y) * 2;
      let DST_S = i32(globalId.z) * 2;
      if (DST_X >= uniforms.outShape[2] || DST_Y >= uniforms.outShape[1] || DST_S >= uniforms.dstSlice) {
        return;
      }
      var r_w0_h0_s0 = vec4<f32>(0.0);
      var r_w1_h0_s0 = vec4<f32>(0.0);
      var r_w0_h1_s0 = vec4<f32>(0.0);
      var r_w1_h1_s0 = vec4<f32>(0.0);
      var r_w0_h0_s1 = vec4<f32>(0.0);
      var r_w1_h0_s1 = vec4<f32>(0.0);
      var r_w0_h1_s1 = vec4<f32>(0.0);
      var r_w1_h1_s1 = vec4<f32>(0.0);
      let xc0 = (DST_X + 0) * uniforms.stride[1] - uniforms.pad[1];
      let xc1 = (DST_X + 1) * uniforms.stride[1] - uniforms.pad[1];
      let yc0 = (DST_Y + 0) * uniforms.stride[0] - uniforms.pad[0];
      let yc1 = (DST_Y + 1) * uniforms.stride[0] - uniforms.pad[0];
      var weights_cache = 0;
      var filters_loc = DST_S * 4 * uniforms.srcSlice * uniforms.filterDims[0] * uniforms.filterDims[1];
      for (var ky = 0; ky < uniforms.filterDims[0]; ky = ky + 1) {
        var yck0 = ky * uniforms.dilation[0] + yc0;
        let in_y0 = yck0 >= 0 && yck0 < uniforms.xShape[1];
        yck0 = clamp(yck0, 0, uniforms.xShape[1] - 1);
        var yck1 = ky * uniforms.dilation[0] + yc1;
        let in_y1 = yck1 >= 0 && yck1 < uniforms.xShape[1];
        yck1 = clamp(yck1, 0, uniforms.xShape[1] - 1);
        for (var kx = 0; kx < uniforms.filterDims[1]; kx = kx + 1) {
          var xck0 = kx * uniforms.dilation[1] + xc0;
          let in_x0 = xck0 >= 0 && xck0 < uniforms.xShape[2];
          xck0 = clamp(xck0, 0, uniforms.xShape[2] - 1);
          var xck1 = kx * uniforms.dilation[1] + xc1;
          let in_x1 = xck1 >= 0 && xck1 < uniforms.xShape[2];
          xck1 = clamp(xck1, 0, uniforms.xShape[2] - 1);
          var addr_w0_h0 = yck0 * uniforms.xShape[2] + xck0;
          var addr_w1_h0 = yck0 * uniforms.xShape[2] + xck1;
          var addr_w0_h1 = yck1 * uniforms.xShape[2] + xck0;
          var addr_w1_h1 = yck1 * uniforms.xShape[2] + xck1;
          let ds = uniforms.sliceStride;

          for (var s = 0; s < uniforms.srcSlice; s = s + 1)
          {
        weights_cache = filters_loc;
        let src_w0_h0 = X.numbers[addr_w0_h0] * f32(in_x0 && in_y0);
        addr_w0_h0 = addr_w0_h0 + ds;
        let src_w1_h0 = X.numbers[addr_w1_h0] * f32(in_x1 && in_y0);
        addr_w1_h0 = addr_w1_h0 + ds;
        let src_w0_h1 = X.numbers[addr_w0_h1] * f32(in_x0 && in_y1);
        addr_w0_h1 = addr_w0_h1 + ds;
        let src_w1_h1 = X.numbers[addr_w1_h1] * f32(in_x1 && in_y1);
        addr_w1_h1 = addr_w1_h1 + ds;

        r_w0_h0_s0.x = r_w0_h0_s0.x + dot(W.numbers[weights_cache + 0], src_w0_h0);
        r_w1_h0_s0.x = r_w1_h0_s0.x + dot(W.numbers[weights_cache + 0], src_w1_h0);
        r_w0_h1_s0.x = r_w0_h1_s0.x + dot(W.numbers[weights_cache + 0], src_w0_h1);
        r_w1_h1_s0.x = r_w1_h1_s0.x + dot(W.numbers[weights_cache + 0], src_w1_h1);
        r_w0_h0_s0.y = r_w0_h0_s0.y + dot(W.numbers[weights_cache + 1], src_w0_h0);
        r_w1_h0_s0.y = r_w1_h0_s0.y + dot(W.numbers[weights_cache + 1], src_w1_h0);
        r_w0_h1_s0.y = r_w0_h1_s0.y + dot(W.numbers[weights_cache + 1], src_w0_h1);
        r_w1_h1_s0.y = r_w1_h1_s0.y + dot(W.numbers[weights_cache + 1], src_w1_h1);
        r_w0_h0_s0.z = r_w0_h0_s0.z + dot(W.numbers[weights_cache + 2], src_w0_h0);
        r_w1_h0_s0.z = r_w1_h0_s0.z + dot(W.numbers[weights_cache + 2], src_w1_h0);
        r_w0_h1_s0.z = r_w0_h1_s0.z + dot(W.numbers[weights_cache + 2], src_w0_h1);
        r_w1_h1_s0.z = r_w1_h1_s0.z + dot(W.numbers[weights_cache + 2], src_w1_h1);
        r_w0_h0_s0.w = r_w0_h0_s0.w + dot(W.numbers[weights_cache + 3], src_w0_h0);
        r_w1_h0_s0.w = r_w1_h0_s0.w + dot(W.numbers[weights_cache + 3], src_w1_h0);
        r_w0_h1_s0.w = r_w0_h1_s0.w + dot(W.numbers[weights_cache + 3], src_w0_h1);
        r_w1_h1_s0.w = r_w1_h1_s0.w + dot(W.numbers[weights_cache + 3], src_w1_h1);
        r_w0_h0_s1.x = r_w0_h0_s1.x + dot(W.numbers[weights_cache + 4], src_w0_h0);
        r_w1_h0_s1.x = r_w1_h0_s1.x + dot(W.numbers[weights_cache + 4], src_w1_h0);
        r_w0_h1_s1.x = r_w0_h1_s1.x + dot(W.numbers[weights_cache + 4], src_w0_h1);
        r_w1_h1_s1.x = r_w1_h1_s1.x + dot(W.numbers[weights_cache + 4], src_w1_h1);
        r_w0_h0_s1.y = r_w0_h0_s1.y + dot(W.numbers[weights_cache + 5], src_w0_h0);
        r_w1_h0_s1.y = r_w1_h0_s1.y + dot(W.numbers[weights_cache + 5], src_w1_h0);
        r_w0_h1_s1.y = r_w0_h1_s1.y + dot(W.numbers[weights_cache + 5], src_w0_h1);
        r_w1_h1_s1.y = r_w1_h1_s1.y + dot(W.numbers[weights_cache + 5], src_w1_h1);
        r_w0_h0_s1.z = r_w0_h0_s1.z + dot(W.numbers[weights_cache + 6], src_w0_h0);
        r_w1_h0_s1.z = r_w1_h0_s1.z + dot(W.numbers[weights_cache + 6], src_w1_h0);
        r_w0_h1_s1.z = r_w0_h1_s1.z + dot(W.numbers[weights_cache + 6], src_w0_h1);
        r_w1_h1_s1.z = r_w1_h1_s1.z + dot(W.numbers[weights_cache + 6], src_w1_h1);
        r_w0_h0_s1.w = r_w0_h0_s1.w + dot(W.numbers[weights_cache + 7], src_w0_h0);
        r_w1_h0_s1.w = r_w1_h0_s1.w + dot(W.numbers[weights_cache + 7], src_w1_h0);
        r_w0_h1_s1.w = r_w0_h1_s1.w + dot(W.numbers[weights_cache + 7], src_w0_h1);
        r_w1_h1_s1.w = r_w1_h1_s1.w + dot(W.numbers[weights_cache + 7], src_w1_h1);
        filters_loc = filters_loc + 8;
          }
        }
      }
      weights_cache = DST_S;
      if (DST_S + 0 >= uniforms.dstSlice)
      {
        return;
      }
      {
       let bias_val = ${
        this.addBias ? 'bias.numbers[weights_cache + 0]' : '0.0'};
      {
        let res = r_w0_h0_s0 + bias_val;
        writeResult(DST_X + 0, DST_Y + 0, DST_S + 0, res);
      }
      if (DST_X + 1 < uniforms.outShape[2]) {
        let res = r_w1_h0_s0 + bias_val;
        writeResult(DST_X + 1, DST_Y + 0, DST_S + 0, res);
      }
      if (DST_Y + 1 < uniforms.outShape[1]) {
        let res = r_w0_h1_s0 + bias_val;
        writeResult(DST_X + 0, DST_Y + 1, DST_S + 0, res);
      }
      if (DST_X + 1 < uniforms.outShape[2] && DST_Y + 1 < uniforms.outShape[1]) {
        let res = r_w1_h1_s0 + bias_val;
        writeResult(DST_X + 1, DST_Y + 1, DST_S + 0, res);
      }
      }
      if (DST_S + 1 >= uniforms.dstSlice)
      {
        return;
      }
      {
        let bias_val = ${
        this.addBias ? 'bias.numbers[weights_cache + 1]' : '0.0'};
      {
        let res = r_w0_h0_s1 + bias_val;
        writeResult(DST_X + 0, DST_Y + 0, DST_S + 1, res);
      }
      if (DST_X + 1 < uniforms.outShape[2]) {
        let res = r_w1_h0_s1 + bias_val;
        writeResult(DST_X + 1, DST_Y + 0, DST_S + 1, res);
      }
      if (DST_Y + 1 < uniforms.outShape[1]) {
        let res = r_w0_h1_s1 + bias_val;
        writeResult(DST_X + 0, DST_Y + 1, DST_S + 1, res);
      }
      if (DST_X + 1 < uniforms.outShape[2] && DST_Y + 1 < uniforms.outShape[1]) {
        let res = r_w1_h1_s1 + bias_val;
        writeResult(DST_X + 1, DST_Y + 1, DST_S + 1, res);
      }
      }

      }
    `;
    return userCode;
  }
}
