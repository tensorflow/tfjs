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
      `filterDims : vec2<u32>; pad : vec2<u32>; stride : vec2<u32>; dilation : vec2<u32>;`;
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
            `fn activation(a : f32, outCoord : vec4<u32>) -> f32{
               let b = getPreluActivationWeightsAtOutCoordsByCoords(outCoord);
               ${activationOp}
             }`;
      } else {
        activationSnippet = `
                  fn activation(a : f32, outCoord : vec4<u32>) -> f32{
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
      fn readInp(batch : u32, row : u32, col : u32, chan : u32) -> f32 {
        let coord = vec4<u32>(batch, row, col, chan);
        if(coordsInBounds4D(coord, uniforms.xShape)) {
          return getX(batch, row, col, chan);
        }
        return 0.0;
      }

      fn readFilt(row : u32, col : u32, xChannel : u32, outChannel : u32) -> f32{
        let coord = vec4<u32>(row, col, xChannel, outChannel);
        if(coordsInBounds4D(coord, uniforms.wShape)) {
          return getW(row, col, xChannel, outChannel);
        }
        return 0.0;
      }

      fn writeResult(batch : u32, row : u32, col : u32, chan : u32, value : f32) {
        let coord = vec4<u32>(batch, row, col, chan);
        if (coordsInBounds4D(coord, uniforms.outShape)) {
          ${addBiasSnippet}
          ${applyActivationSnippet}
          setOutput(batch, row, col, chan, value);
        }
      }

      ${getMainHeaderStringWgsl(this.workGroupSize)} {
        ${getGlobalIndexStringWgsl(this.workGroupSize)}
        let coords = getOutputCoords(globalId, index);
        let batch = coords[0];
        let outChannel = coords[3];

        var acc = 0.0;

        for (var row = 0u; row < uniforms.filterDims[0]; row = row + 1u) {
          for (var col = 0u; col < uniforms.filterDims[1]; col = col + 1u) {
            for (var xChannel = 0u; xChannel < uniforms.xShape[3]; xChannel = xChannel + 1u) {
              let coordRow = i32(coords[1] * uniforms.stride[0] + uniforms.dilation[0] * row - uniforms.pad[0]);
              if (coordRow < 0) {
                continue;
              }
              let coordCol = i32(coords[2] * uniforms.stride[1] + uniforms.dilation[1] * col - uniforms.pad[1]);
              if (coordCol < 0) {
                continue;
              }
              let v = readInp(batch, u32(coordRow), u32(coordCol), xChannel);
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
