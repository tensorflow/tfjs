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

import * as tf from '@tensorflow/tfjs-core';
import {Conv2DInfo} from '@tensorflow/tfjs-core/dist/ops/conv_util';

import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class Conv2DNaiveProgram implements WebGPUProgram {
  outputShape: number[];
  userCode: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride;';
  workGroupSize: [number, number, number] = [4, 8, 1];

  constructor(convInfo: Conv2DInfo) {
    this.outputShape = convInfo.outShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0, 3]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    tf.util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');
    tf.util.assert(
        convInfo.dilationHeight === 1 && convInfo.dilationWidth === 1,
        () => 'TODO: Dilation is unimplemented');

    this.userCode = `
      bool coordIsValid(ivec4 coord, ivec4 shape) {
        return all(greaterThanEqual(coord, ivec4(0))) &&
            all(lessThan(coord, shape));
      }

      float readInp(uint batch, uint row, uint col, uint chan) {
        ivec4 coord = ivec4(batch, row, col, chan);
        return coordIsValid(coord, xShape) ? x[getFlatIndex(coord, xShape)] : 0;
      }

      float readFilt(uint row, uint col, uint xChannel, uint outChannel) {
        ivec4 coord = ivec4(row, col, xChannel, outChannel);
        ivec4 shape = ivec4(filterDims, xShape[3], outShape[3]);
        return coordIsValid(coord, shape) ? W[getFlatIndex(coord, shape)] : 0;
      }

      void writeResult(uint batch, uint row, uint col, uint chan, float value) {
        ivec4 coord = ivec4(batch, row, col, chan);
        if (coordIsValid(coord, outShape)) {
          result[getFlatIndex(coord, outShape)] = value;
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
                  pad[0] + coords[1] * stride[0] + row,
                  pad[1] + coords[2] * stride[1] + col, xChannel);
              float f = readFilt(row, col, xChannel, outChannel);
              acc += v * f;
            }
          }
        }

        writeResult(batch, coords[1], coords[2], outChannel, acc);
      }
    `;
  }
}
