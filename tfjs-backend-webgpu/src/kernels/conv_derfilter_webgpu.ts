/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
import {computeDispatch} from '../webgpu_util';
import {WebGPUProgram} from './webgpu_program';

export class Conv2DDerFilterProgram implements WebGPUProgram {
  variableNames = ['x', 'dy'];
  outputShape: number[];
  userCode: string;
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number];

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.filterShape;

    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const isChannelsLast = convInfo.dataFormat === 'channelsLast';
    this.outputShape = convInfo.outShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0, 3]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    this.userCode = `
    void writeResult(int batch, int row, int col, int chan, float value) {
      ivec4 coord = ivec4(batch, row, col, chan);
      if (coordsInBounds(coord, outShape)) {
        setOutput(batch, row, col, chan, value);
      }
    }
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int d2 = coords.w;

        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int b = 0; b < ${convInfo.batchSize}; b++) {
          for (int yR = 0; yR < ${convInfo.outHeight}; yR++) {
            int xR = wR + yR * ${strideHeight} - ${padTop};

            if (xR < 0 || xR >= ${convInfo.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${convInfo.outWidth}; yC++) {
              int xC = wC + yC * ${strideWidth} - ${padLeft};

              if (xC < 0 || xC >= ${convInfo.inWidth}) {
                continue;
              }

              if (${isChannelsLast}) {
                float dyValue = getDy(b, yR, yC, d2);
                float xValue = getX(b, xR, xC, d1);
                dotProd += (xValue * dyValue);
              } else {
                float dyValue = getDy(b, d2, yR, yC);
                float xValue = getX(b, d1, xR, xC);
                dotProd += (xValue * dyValue);
              }

            }
          }
        }
        writeResult(coords[0], coords[1], coords[2], coords[3], dotProd);
      }
    `;
    this.shaderKey = `conv2dderfilter`;
  }
}
