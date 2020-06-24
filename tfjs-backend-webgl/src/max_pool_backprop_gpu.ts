/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
import {GPGPUProgram} from './gpgpu_math';

export class MaxPool2DBackpropProgram implements GPGPUProgram {
  variableNames = ['dy', 'maxPos'];
  outputShape: number[];
  userCode: string;

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.inShape;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;

    const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
    const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;

    const lastIndex = effectiveFilterHeight * effectiveFilterWidth - 1;
    this.userCode = `
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${effectiveFilterHeight};
          wR += ${dilationHeight}) {
          float dyR = float(dyRCorner + wR) / ${strideHeight}.0;

          if (dyR < 0.0 || dyR >= ${convInfo.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${effectiveFilterWidth}; wC++) {
            float dyC = float(dyCCorner + wC) / ${strideWidth}.0;

            if (dyC < 0.0 || dyC >= ${convInfo.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);
            int maxPosValue = ${lastIndex} - int(getMaxPos(b, idyR, idyC, d));

            // Get the current value, check it against the value from the
            // position matrix.
            int curPosValue = wR * ${effectiveFilterWidth} + wC;
            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

            dotProd += dyValue * mask;
          }
        }
        setOutput(dotProd);
      }
    `;
  }
}

export class MaxPool3DBackpropProgram implements GPGPUProgram {
  variableNames = ['dy', 'maxPos'];
  outputShape: number[];
  userCode: string;

  constructor(convInfo: backend_util.Conv3DInfo) {
    this.outputShape = convInfo.inShape;
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;

    const padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
    const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
    const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;

    const lastIndex =
        effectiveFilterDepth * effectiveFilterHeight * effectiveFilterWidth - 1;
    this.userCode = `
      const ivec3 pads = ivec3(${padFront}, ${padTop}, ${padLeft});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${effectiveFilterDepth};
           wD += ${dilationDepth}) {
          float dyD = float(dyDCorner + wD) / ${strideDepth}.0;

          if (dyD < 0.0 || dyD >= ${convInfo.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${effectiveFilterHeight};
              wR += ${dilationHeight}) {
            float dyR = float(dyRCorner + wR) / ${strideHeight}.0;

            if (dyR < 0.0 || dyR >= ${convInfo.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${effectiveFilterWidth};
                wC += ${dilationWidth}) {
              float dyC = float(dyCCorner + wC) / ${strideWidth}.0;

              if (dyC < 0.0 || dyC >= ${convInfo.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);
              int maxPosValue = ${lastIndex} -
                  int(getMaxPos(batch, idyD, idyR, idyC, ch));

              // Get the current value, check it against the value from the
              // position matrix.
              int curPosValue =
                  wD * ${effectiveFilterHeight} * ${effectiveFilterWidth} +
                  wR * ${effectiveFilterWidth} + wC;
              float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

              dotProd += dyValue * mask;
            }
          }
        }
        setOutput(dotProd);
      }
    `;
  }
}
