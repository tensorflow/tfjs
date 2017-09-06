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

export class Conv2DDerWeightsProgram implements GPGPUProgram {
  variableNames = ['x', 'dy'];
  params: Array<{}>;
  outputShape: number[];
  userCode: string;

  constructor(
      xShape: [number, number, number], fSize: number, outputDepth: number,
      stride: number, zeroPad: number) {
    const yShape = conv_util.computeOutputShape3D(
        xShape, fSize, outputDepth, stride, zeroPad);
    const yNumRows = yShape[0];
    const yNumCols = yShape[1];
    const xNumRows = xShape[0];
    const xNumCols = xShape[1];
    this.outputShape =
        conv_util.computeWeightsShape4D(xShape[2], outputDepth, fSize);
    this.params = [stride, zeroPad];
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
        for (int yR = 0; yR < ${yNumRows}; yR++) {
          int xR = wR + yR * ${stride} - ${zeroPad};

          if (xR < 0 || xR >= ${xNumRows}) {
            continue;
          }

          for (int yC = 0; yC < ${yNumCols}; yC++) {
            int xC = wC + yC * ${stride} - ${zeroPad};

            if (xC < 0 || xC >= ${xNumCols}) {
              continue;
            }

            float dyValue = getDy(yR, yC, d2);
            float xValue = getX(xR, xC, d1);
            dotProd += (xValue * dyValue);
          }
        }
        setOutput(dotProd);
      }
    `;
  }
}

export class Conv2DTransposeProgram implements GPGPUProgram {
  variableNames = ['x', 'W', 'bias'];
  params: Array<{}>;
  outputShape: number[];
  userCode: string;

  constructor(
      xShape: [number, number, number], fSize: number, origInputDepth: number,
      origStride: number, origPad: number, hasBias: boolean) {
    const [xRows, xCols, origOutputDepth] = xShape;
    const biasSnippet = hasBias ? 'dotProd += getBias(d2);' : '';

    // Figure out the output shape by dilating the input.
    const xRowsDilated = (xRows - 1) * origStride + 1;
    const xColsDilated = (xCols - 1) * origStride + 1;
    const pad = fSize - 1 - origPad;
    this.outputShape = conv_util.computeOutputShape3D(
        [xRowsDilated, xColsDilated, origOutputDepth], fSize, origInputDepth, 1,
        pad);
    this.params = [pad, fSize, origStride, hasBias];

    this.userCode = `
      const ivec2 pads = ivec2(${pad}, ${pad});

      void main() {
        ivec3 coords = getOutputCoords();
        int d2 = coords.z;

        ivec2 xRCCorner = coords.xy - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d2, d1) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${fSize}; wR++) {
          float xR = float(xRCorner + wR) / ${origStride}.0;

          if (xR < 0.0 || xR >= ${xRows}.0 || fract(xR) > 0.0) {
            continue;
          }
          int ixR = int(xR);

          int wRPerm = ${fSize} - 1 - wR;

          for (int wC = 0; wC < ${fSize}; wC++) {
            float xC = float(xCCorner + wC) / ${origStride}.0;

            if (xC < 0.0 || xC >= ${xCols}.0 || fract(xC) > 0.0) {
              continue;
            }
            int ixC = int(xC);

            int wCPerm = ${fSize} - 1 - wC;

            for (int d1 = 0; d1 < ${origOutputDepth}; d1++) {
              float xValue = getX(ixR, ixC, d1);
              float wValue = getW(wRPerm, wCPerm, d2, d1);
              dotProd += xValue * wValue;
            }
          }
        }
        ${biasSnippet}
        setOutput(dotProd);
      }
    `;
  }
}

export class Conv2DDerBiasProgram implements GPGPUProgram {
  variableNames = ['dy'];
  params: Array<{}> = [];
  outputShape: number[];
  userCode: string;

  constructor(yShape: [number, number, number]) {
    const [yNumRows, yNumCols, outputDepth] = yShape;
    this.outputShape = [outputDepth];
    this.userCode = `
      void main() {
        int d2 = getOutputCoords();

        float derBias = 0.0;
        for (int yR = 0; yR < ${yNumRows}; yR++) {
          for (int yC = 0; yC < ${yNumCols}; yC++) {
            derBias += getDy(yR, yC, d2);
          }
        }
        setOutput(derBias);
      }
    `;
  }
}
