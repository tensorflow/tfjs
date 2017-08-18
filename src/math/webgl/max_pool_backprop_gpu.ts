/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as conv_util from '../conv_util';

import {GPGPUProgram} from './gpgpu_math';

export class MaxPool2DBackpropProgram implements GPGPUProgram {
  variableNames = ['dy', 'maxPos'];
  params: Array<{}>;
  outputShape: number[];
  userCode: string;

  constructor(
      dyShape: [number, number, number], fSize: number, origStride: number,
      origPad: number) {
    const pad = fSize - 1 - origPad;
    const dyRows = dyShape[0];
    const dyCols = dyShape[1];
    this.params = [fSize, origStride, origPad];

    const dilatedDyRC =
        conv_util.computeDilatedRC([dyShape[0], dyShape[1]], origStride);
    this.outputShape = conv_util.computeOutputShape3D(
        [dilatedDyRC[0], dilatedDyRC[1], dyShape[2]], fSize, dyShape[2], 1,
        pad);

    this.userCode = `
      void main() {
        vec3 coords = getOutputCoords();
        float dxR = coords.x;
        float dxC = coords.y;
        float d = coords.z;

        vec2 dyRCCorner = vec2(dxR, dxC) - vec2(${pad}.0, ${pad}.0);
        float dyRCorner = dyRCCorner.x;
        float dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(yR, dxC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int iwR = 0; iwR < ${fSize}; iwR++) {
          float wR = float(iwR);
          float dyR = (dyRCorner + wR) / ${origStride}.0;

          if (dyR < 0.0 || dyR >= ${dyRows}.0 || fract(dyR) > 0.0) {
            continue;
          }

          for (int iwC = 0; iwC < ${fSize}; iwC++) {
            float wC = float(iwC);
            float dyC = (dyCCorner + wC) / ${origStride}.0;

            if (dyC < 0.0 || dyC >= ${dyCols}.0 || fract(dyC) > 0.0) {
              continue;
            }

            float dyValue = getDy(dyR, dyC, d);
            float maxPosValue =
                ${fSize * fSize - 1}.0 - getMaxPos(dyR, dyC, d);

            // Get the current value, check it against the value from the
            // position matrix.
            float curPosValue = wR * ${fSize}.0 + wC;
            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

            dotProd += dyValue * mask;
          }
        }
        setOutput(dotProd);
      }
    `;
  }
}
