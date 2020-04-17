/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
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

export class Conv2DDerFilterProgram implements GPGPUProgram {
  variableNames = ['x', 'dy'];
  outputShape: number[];
  userCode: string;

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.filterShape;

    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const isChannelsLast = convInfo.dataFormat === 'channelsLast';

    this.userCode = `
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
        setOutput(dotProd);
      }
    `;
  }
}

export class Conv2DDerInputProgram implements GPGPUProgram {
  variableNames = ['dy', 'W'];
  outputShape: number[];
  userCode: string;

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.inShape;

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

          if (dyR < 0.0 || dyR >= ${convInfo.outHeight}.0 || fract(dyR) > 0.0) {
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
        setOutput(dotProd);
      }
    `;
  }
}

export class Conv3DDerFilterProgram implements GPGPUProgram {
  variableNames = ['x', 'dy'];
  outputShape: number[];
  userCode: string;

  constructor(convInfo: backend_util.Conv3DInfo) {
    this.outputShape = convInfo.filterShape;

    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const padFront = convInfo.padInfo.front;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;

    this.userCode = `
      void main() {
        ivec5 coords = getOutputCoords();
        int wF = coords.x;
        int wR = coords.y;
        int wC = coords.z;
        int d1 = coords.w;
        int d2 = coords.u;

        float dotProd = 0.0;

        for (int b = 0; b < ${convInfo.batchSize}; b++) {
          for (int yF = 0; yF < ${convInfo.outDepth}; yF++) {
            int xF = wF + yF * ${strideDepth} - ${padFront};

            if (xF < 0 || xF >= ${convInfo.inDepth}) {
              continue;
            }

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

                float dyValue = getDy(b, yF, yR, yC, d2);
                float xValue = getX(b, xF, xR, xC, d1);
                dotProd += (xValue * dyValue);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `;
  }
}

export class Conv3DDerInputProgram implements GPGPUProgram {
  variableNames = ['dy', 'W'];
  outputShape: number[];
  userCode: string;

  constructor(convInfo: backend_util.Conv3DInfo) {
    this.outputShape = convInfo.inShape;

    const filterDepth = convInfo.filterDepth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;

    const padFront = filterDepth - 1 - convInfo.padInfo.front;
    const padTop = filterHeight - 1 - convInfo.padInfo.top;
    const padLeft = filterWidth - 1 - convInfo.padInfo.left;

    this.userCode = `
      const ivec3 pads = ivec3(${padFront}, ${padTop}, ${padLeft});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.u;


        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyFCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        float dotProd = 0.0;
        for (int wF = 0; wF < ${filterDepth}; wF++) {
          float dyF = float(dyFCorner + wF) / ${strideDepth}.0;

          if (dyF < 0.0 || dyF >= ${convInfo.outDepth}.0 || fract(dyF) > 0.0) {
            continue;
          }
          int idyF = int(dyF);

          int wFPerm = ${filterDepth} - 1 - wF;

          for (int wR = 0; wR < ${filterHeight}; wR++) {
            float dyR = float(dyRCorner + wR) / ${strideHeight}.0;

            if (dyR < 0.0 || dyR >= ${convInfo.outHeight}.0 ||
              fract(dyR) > 0.0) {
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
                float xValue = getDy(batch, idyF, idyR, idyC, d2);
                float wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `;
  }
}
