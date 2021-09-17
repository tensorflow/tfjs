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

import {getGlobalIndexStringWgsl, getMainHeaderStringWgsl} from '../shader_preprocessor_wgsl';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

import {mapActivationToShaderProgram} from './activation_util';
import {getUseWgsl, WebGPUProgram} from './webgpu_program';

export class Conv2DNaiveProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation;';
  uniformsWgsl =
      `filterDims : vec2<i32>; pad : vec2<i32>; stride : vec2<i32>; dilation : vec2<i32>;`;
  workGroupSize: [number, number, number] = [128, 1, 1];
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  useWgsl: boolean;

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: backend_util.Activation = null,
      hasPreluActivationWeights = false) {
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

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.convInfo = convInfo;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;

    this.shaderKey = `conv2DNaive_${this.activation}`;
    this.useWgsl = getUseWgsl();
  }

  getUserCode(): string {
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(this.activation);
      if (this.hasPreluActivationWeights) {
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

      applyActivationSnippet = `value = activation(value);`;
    }

    const addBiasSnippet = this.addBias ? 'value += getBiasAtOutCoords();' : '';

    const userCode = `
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
                  coords[1] * stride[0] + dilation[0] * row - pad[0],
                  coords[2] * stride[1] + dilation[1] * col - pad[1],
                  xChannel);
              float f = readFilt(row, col, xChannel, outChannel);
              acc += v * f;
            }
          }
        }

        writeResult(batch, coords[1], coords[2], outChannel, acc);
      }
    `;
    return userCode;
  }

  getUserCodeWgsl(): string {
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(this.activation);
      if (this.hasPreluActivationWeights) {
        activationSnippet =
            `fn activation(a : f32, outCoord : vec4<i32>) -> f32{
               let b = getPreluActivationWeightsAtOutCoordsByCoords(outCoord);
               ${activationOp}
             }`;
      } else {
        activationSnippet = `
                  fn activation(a : f32, outCoord : vec4<i32>) -> f32{
                    ${activationOp}
                  }
                `;
      }

      applyActivationSnippet = `value = activation(value, outCoord);`;
    }

    const addBiasSnippet = this.addBias ?
        'value = value + getBiasAtOutCoordsByCoords(outCoord);' :
        '';

    const userCode = `
      ${activationSnippet}
      fn readInp(batch : i32, row : i32, col : i32, chan : i32) -> f32 {
        let coord = vec4<i32>(batch, row, col, chan);
        if(coordsInBounds4D(coord, uniforms.xShape)) {
          return getX(batch, row, col, chan);
        }
        return 0.0;
      }

      fn readFilt(row : i32, col : i32, xChannel : i32, outChannel : i32) -> f32{
        let coord = vec4<i32>(row, col, xChannel, outChannel);
        if(coordsInBounds4D(coord, uniforms.wShape)) {
          return getW(row, col, xChannel, outChannel);
        }
        return 0.0;
      }

      fn writeResult(batch : i32, row : i32, col : i32, chan : i32, value : f32) {
        let coord = vec4<i32>(batch, row, col, chan);
        if (coordsInBounds4D(coord, uniforms.outShape)) {
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutput(batch, row, col, chan, value);
        }
      }

      ${getMainHeaderStringWgsl()} {
        ${getGlobalIndexStringWgsl()}
        let coords = getOutputCoords(globalId, index);
        let batch = coords[0];
        let outChannel = coords[3];

        var acc = 0.0;

        for (var row = 0; row < uniforms.filterDims[0]; row = row + 1) {
          for (var col = 0; col < uniforms.filterDims[1]; col = col + 1) {
            for (var xChannel = 0; xChannel < uniforms.xShape[3]; xChannel = xChannel + 1) {
              let coordRow = coords[1] * uniforms.stride[0] + uniforms.dilation[0] * row - uniforms.pad[0];
              let coordCol = coords[2] * uniforms.stride[1] + uniforms.dilation[1] * col - uniforms.pad[1];
              let v = readInp(batch, coordRow, coordCol, xChannel);
              let f = readFilt(row, col, xChannel, outChannel);
              acc = acc + v * f;
            }
          }
        }

        writeResult(batch, coords[1], coords[2], outChannel, acc);
      }
    `;
    return userCode;
  }
}
