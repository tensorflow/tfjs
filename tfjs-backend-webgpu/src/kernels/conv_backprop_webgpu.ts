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

import {backend_util} from '@tensorflow/tfjs-core';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
import {WebGPUProgram} from './webgpu_program';

export class Conv2DDerInputProgram implements WebGPUProgram {
  variableNames = ['dy', 'W'];
  uniforms = 'ivec2 filterDims, pads, stride; ivec4 outBackprop;';
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number] = [64, 1, 1];
  isChannelsLast: boolean;

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.inShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
    this.shaderKey = `conv2DDerInput_${this.isChannelsLast}`;
  }

  getUserCode(): string {
    const rowDim = this.isChannelsLast ? 1 : 2;
    const colDim = this.isChannelsLast ? 2 : 3;
    const channelDim = this.isChannelsLast ? 3 : 1;
    return `
    void main() {
      ivec4 coords = getOutputCoords();
      if (coordsInBounds(coords, outShape)) {
        int batch = coords[0];
        int d1 = coords[${channelDim}];

        ivec2 dyCorner = ivec2(coords[${rowDim}], coords[${colDim}]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < filterDims.x; wR++) {
          float dyR = float(dyRCorner + wR) / float(stride.x);

          if (dyR < 0.0 || dyR >= float(outBackprop[1]) || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = filterDims.x - 1 - wR;

          for (int wC = 0; wC < filterDims.y; wC++) {
            float dyC = float(dyCCorner + wC) / float(stride.y);

            if (dyC < 0.0 || dyC >= float(outBackprop[2]) ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = filterDims.y - 1 - wC;

            for (int d2 = 0; d2 < outBackprop[3]; d2++) {

              if (${this.isChannelsLast}) {
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
        setOutput(coords[0], coords[1], coords[2], coords[3], dotProd);
      }
    }
  `;
  }
}
