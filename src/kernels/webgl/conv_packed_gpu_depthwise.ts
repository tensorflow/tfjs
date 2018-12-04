/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import {Conv2DInfo} from '../../ops/conv_util';
import {GPGPUProgram} from './gpgpu_math';

export class DepthwiseConvPacked2DProgram implements GPGPUProgram {
  variableNames = ['x', 'W'];
  usesPackedTextures = true;
  outputShape: number[];
  userCode: string;

  constructor(convInfo: Conv2DInfo) {
    this.outputShape = convInfo.outShape;

    const xNumRows = convInfo.inHeight;
    const xNumCols = convInfo.inWidth;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const texelsAcross = Math.ceil((filterWidth + 1) / 2);

    let mainLoop = `int xR; int xC;`;

    for (let r = 0; r < filterHeight; r++) {
      for (let c = -padLeft; c < texelsAcross * 2; c++) {
        mainLoop += `vec4 ${xTexelName(r, c)} = vec4(0.);`;
      }

      for (let c = 0; c < filterWidth; c++) {
        mainLoop += `
          vec4 wR${r}C${c} = vec4(0.);
          vec4 xR${r}C${c} = vec4(0.);`;
      }
    }

    /**
     * This vectorized implementation of depthwiseConv works by gathering the
     * values needed for each output channel's dot product into vec4's and then
     * multiplying them all together (this happens in the final double for-loop
     * below). Most of the main loop consists of constructing these vec4's with
     * the minimum number of texture2D calls as possible, which entails logic
     * for making use of all four returned values from a texture2D call at once.
     */
    for (let r = 0; r < filterHeight; r++) {
      for (let c = 0; c < texelsAcross; c++) {
        const col = c * 2;
        const left = c * 2 + padLeft;

        mainLoop += `
          xR = xRCorner + ${r};
          xC = xCCorner + ${left};

          if(xR >= 0 && xR < ${xNumRows} && xC >= 0 && xC < ${xNumCols}) {
            ${xTexelName(r, left)} = getX(batch, xR, xC, d1);
          }`;

        if (padLeft === 0) {
          if (col < filterWidth && c === texelsAcross - 1) {
            if (strideWidth > 1) {
              mainLoop += `
                vec4 ${xTexelName(r, left + 2)} = vec4(0.);

                if(xR >= 0 && xR < ${xNumRows} && xC + 2 < ${xNumCols}) {
                  ${xTexelName(r, left + 2)} = getX(batch, xR, xC + 2, d1);
                }`;
            }

            mainLoop += `
              xR${r}C${left} = ${constructTexel(r, left, strideWidth, padLeft)};
            `;
          }
        } else if (c === 0) {
          mainLoop += `
            if(xR >= 0 && xR < ${xNumRows} && xC - 2 >= 0) {
              ${xTexelName(r, left - 2)} = getX(batch, xR, xC - 2, d1);
            }`;
        }

        if (col > 0) {
          mainLoop += `xR${r}C${left - 2} =
            ${constructTexel(r, left - 2, strideWidth, padLeft)};`;
        }

        if (left - 1 >= 0 && left - 1 < filterWidth) {
          mainLoop += `xR${r}C${left - 1} =
              ${constructTexel(r, left - 1, strideWidth, padLeft)};`;
        }

        if (col < filterWidth) {
          mainLoop += `
            vec4 wTexel${r}C${col} = getW(${r}, ${col}, d1, q);
            wR${r}C${col} = vec4(wTexel${r}C${col}.xz, wTexel${r}C${col}.xz);
          `;

          if (col + 1 < filterWidth) {
            mainLoop += `
              vec4 wTexelR${r}C${col + 1} = getW(${r}, ${col + 1}, d1, q);
              wR${r}C${col + 1} =
                vec4(wTexelR${r}C${col + 1}.xz, wTexelR${r}C${col + 1}.xz);`;
          }
        }
      }
    }

    for (let r = 0; r < filterHeight; r++) {
      for (let c = 0; c < filterWidth; c++) {
        mainLoop += `result += xR${r}C${c} * wR${r}C${c};`;
      }
    }

    this.userCode = `
      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2;
        int q = 0;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        vec4 result = vec4(0.);

        ${mainLoop}

        setOutput(result);
      }
    `;
  }
}

function xTexelName(r: number, c: number): string {
  return `xTexelR${r}C${c < 0 ? 'minus' + Math.abs(c).toString() : c}`;
}

/**
 * Given a 2x2 filter, we want to multiply xR0C0, wR0C0, xR0C1, wR0C1, xR1C0,
 * wR1C0, xR1C1, xR1C1. The xRC's are constructed out of xTexelRC's, which are
 * the vec4's returned from sampling calls. Sometimes this means mixing channels
 * from adjacent samples, which constructTexel handles.
 */
function constructTexel(
    r: number, c: number, stride: number, padLeft: number): string {
  if (stride === 1) {
    if (padLeft % 2 === c % 2) {
      return xTexelName(r, c);
    }
    return `vec4(${xTexelName(r, c - 1)}.zw, ${xTexelName(r, c + 1)}.xy)`;
  }

  if (padLeft % 2 === c % 2) {
    return `vec4(${xTexelName(r, c)}.xy, ${xTexelName(r, c + 2)}.xy)`;
  }
  return `vec4(${xTexelName(r, c - 1)}.zw, ${xTexelName(r, c + 1)}.zw)`;
}
