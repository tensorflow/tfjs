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
import {getMainHeaderString} from './shader_preprocessor';
import {WebGPUProgram} from './webgpu_program';
import {computeDispatch, flatDispatchLayout} from './webgpu_util';

export class Conv2DNaiveProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms =
      `filterDims : vec2<i32>, pad : vec2<i32>, stride : vec2<i32>, dilation : vec2<i32>,`;
  workGroupSize: [number, number, number] = [128, 1, 1];
  convInfo: backend_util.Conv2DInfo;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;

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
  }

  getUserCode(): string {
    let activationSnippet = '', applyActivationSnippet = '';
    if (this.activation) {
      const activationOp = mapActivationToShaderProgram(this.activation);
      if (this.hasPreluActivationWeights) {
        activationSnippet =
            `fn activation(a : f32, outCoord : vec4<i32>) -> f32{
               let b = getPreluActivationWeightsByOutputCoords(outCoord);
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

    const addBiasSnippet =
        this.addBias ? 'value = value + getBiasByOutputCoords(outCoord);' : '';

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
          setOutputAtCoords(batch, row, col, chan, value);
        }
      }

      ${getMainHeaderString()}
        let coords = getOutputCoords();
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
