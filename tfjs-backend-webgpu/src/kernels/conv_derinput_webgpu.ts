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

export class Conv2DDerInputProgram implements WebGPUProgram {
  variableNames = ['dy', 'W'];
  outputShape: number[];
  userCode: string;
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [4, 8, 4];

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.inShape;
    this.dispatchLayout = {x: [2], y: [1], z: [0, 3]};
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const isChannelsLast = convInfo.dataFormat === 'channelsLast';

    const padTop = filterHeight - 1 - convInfo.padInfo.top;
    const padLeft = filterWidth - 1 - convInfo.padInfo.left;

    const rowDim = isChannelsLast ? 1 : 2;
    const colDim = isChannelsLast ? 2 : 3;
    const channelDim = isChannelsLast ? 3 : 1;

    this.userCode = `
      const ivec2 pads = ivec2(${padTop}, ${padLeft});
      void writeResult(int batch, int row, int col, int chan, float value) {
        ivec4 coord = ivec4(batch, row, col, chan);
        if (coordsInBounds(coord, outShape)) {
          setOutput(batch, row, col, chan, value);
        }
      }
      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[${channelDim}];

        ivec2 dyCorner = ivec2(coords[${rowDim}], coords[${colDim}]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${filterHeight}; wR++) {
          float dyR = float(dyRCorner + wR) / ${strideHeight}.0;

          if (dyR < 0.0 || dyR >= ${
        convInfo.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${filterHeight} - 1 - wR;

          for (int wC = 0; wC < ${filterWidth}; wC++) {
            float dyC = float(dyCCorner + wC) / ${strideWidth}.0;

            if (dyC < 0.0 || dyC >= ${convInfo.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${filterWidth} - 1 - wC;

            for (int d2 = 0; d2 < ${convInfo.outChannels}; d2++) {

              if (${isChannelsLast}) {
                float xValue = getDy(batch, idyR, idyC, d2);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              } else {
                float xValue = getDy(batch, d2, idyR, idyC);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }

            }
          }
        }
        writeResult(batch, coords[1], coords[2], coords[3], dotProd);
      }
    `;
    this.shaderKey = `conv2dderinput`;
  }
}
