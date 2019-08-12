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

import {Conv2DInfo, Conv3DInfo} from '../../ops/conv_util';
import {GPGPUProgram} from './gpgpu_math';

export class Conv2DProgram implements GPGPUProgram {
  variableNames = ['x', 'W'];
  outputShape: number[];
  userCode: string;

  constructor(
      convInfo: Conv2DInfo, addBias = false, activation: string = null,
      hasPreluActivationWeights = false) {
    this.outputShape = convInfo.outShape;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;

    const inputDepthNearestVec4 = Math.floor(convInfo.inChannels / 4) * 4;
    const inputDepthVec4Remainder = convInfo.inChannels % 4;
    const isChannelsLast = convInfo.dataFormat === 'channelsLast';

    const rowDim = isChannelsLast ? 1 : 2;
    const colDim = isChannelsLast ? 2 : 3;
    const channelDim = isChannelsLast ? 3 : 1;

    let activationSnippet = '', applyActivationSnippet = '';
    if (activation) {
      if (hasPreluActivationWeights) {
        activationSnippet = `float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${activation}
        }`;
      } else {
        activationSnippet = `
          float activation(float x) {
            ${activation}
          }
        `;
      }

      applyActivationSnippet = `result = activation(result);`;
    }

    const addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
    if (addBias) {
      this.variableNames.push('bias');
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
    }

    this.userCode = `
      ${activationSnippet}

      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d2 = coords[${channelDim}];

        ivec2 xRCCorner =
            ivec2(coords[${rowDim}], coords[${colDim}]) * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${filterHeight}; wR++) {
          int xR = xRCorner + wR * ${dilationHeight};

          if (xR < 0 || xR >= ${convInfo.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${filterWidth}; wC++) {
            int xC = xCCorner + wC * ${dilationWidth};

            if (xC < 0 || xC >= ${convInfo.inWidth}) {
              continue;
            }

            for (int d1 = 0; d1 < ${inputDepthNearestVec4}; d1 += 4) {
              vec4 wValues = vec4(
                getW(wR, wC, d1, d2),
                getW(wR, wC, d1 + 1, d2),
                getW(wR, wC, d1 + 2, d2),
                getW(wR, wC, d1 + 3, d2)
              );

              if (${isChannelsLast}) {
                vec4 xValues = vec4(
                  getX(batch, xR, xC, d1),
                  getX(batch, xR, xC, d1 + 1),
                  getX(batch, xR, xC, d1 + 2),
                  getX(batch, xR, xC, d1 + 3)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec4 xValues = vec4(
                  getX(batch, d1, xR, xC),
                  getX(batch, d1 + 1, xR, xC),
                  getX(batch, d1 + 2, xR, xC),
                  getX(batch, d1 + 3, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }
            }

            if (${inputDepthVec4Remainder === 1}) {

              if (${isChannelsLast}) {
                dotProd +=
                    getX(batch, xR, xC, ${inputDepthNearestVec4}) *
                    getW(wR, wC, ${inputDepthNearestVec4}, d2);
              } else {
                dotProd +=
                    getX(batch, ${inputDepthNearestVec4}, xR, xC) *
                    getW(wR, wC, ${inputDepthNearestVec4}, d2);
              }

            } else if (${inputDepthVec4Remainder === 2}) {
              vec2 wValues = vec2(
                getW(wR, wC, ${inputDepthNearestVec4}, d2),
                getW(wR, wC, ${inputDepthNearestVec4} + 1, d2)
              );

              if (${isChannelsLast}) {
                vec2 xValues = vec2(
                  getX(batch, xR, xC, ${inputDepthNearestVec4}),
                  getX(batch, xR, xC, ${inputDepthNearestVec4} + 1)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec2 xValues = vec2(
                  getX(batch, ${inputDepthNearestVec4}, xR, xC),
                  getX(batch, ${inputDepthNearestVec4} + 1, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            } else if (${inputDepthVec4Remainder === 3}) {
              vec3 wValues = vec3(
                getW(wR, wC, ${inputDepthNearestVec4}, d2),
                getW(wR, wC, ${inputDepthNearestVec4} + 1, d2),
                getW(wR, wC, ${inputDepthNearestVec4} + 2, d2)
              );

              if (${isChannelsLast}) {
                vec3 xValues = vec3(
                  getX(batch, xR, xC, ${inputDepthNearestVec4}),
                  getX(batch, xR, xC, ${inputDepthNearestVec4} + 1),
                  getX(batch, xR, xC, ${inputDepthNearestVec4} + 2)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec3 xValues = vec3(
                  getX(batch, ${inputDepthNearestVec4}, xR, xC),
                  getX(batch, ${inputDepthNearestVec4} + 1, xR, xC),
                  getX(batch, ${inputDepthNearestVec4} + 2, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            }
          }
        }

        float result = dotProd;
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(result);
      }
    `;
  }
}

export class Conv3DProgram implements GPGPUProgram {
  variableNames = ['x', 'W'];
  outputShape: number[];
  userCode: string;

  constructor(convInfo: Conv3DInfo) {
    this.outputShape = convInfo.outShape;
    const padFront = convInfo.padInfo.front;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const filterDepth = convInfo.filterDepth;
    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;

    const inputDepthNearestVec4 = Math.floor(convInfo.inChannels / 4) * 4;
    const inputDepthVec4Remainder = convInfo.inChannels % 4;

    this.userCode = `
      const ivec3 strides = ivec3(${strideDepth}, ${strideHeight}, ${
        strideWidth});
      const ivec3 pads = ivec3(${padFront}, ${padTop}, ${padLeft});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d2 = coords.u;

        ivec3 xFRCCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xFCorner = xFRCCorner.x;
        int xRCorner = xFRCCorner.y;
        int xCCorner = xFRCCorner.z;

        // Convolve x(?, ?, ?, d1) with w(:, :, :, d1, d2) to get
        // y(yF, yR, yC, d2). ? = to be determined. : = across all
        // values in that axis.
        float dotProd = 0.0;
        for (int wF = 0; wF < ${filterDepth}; wF++) {
          int xF = xFCorner + wF * ${dilationDepth};

          if (xF < 0 || xF >= ${convInfo.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${filterHeight}; wR++) {
            int xR = xRCorner + wR * ${dilationHeight};

            if (xR < 0 || xR >= ${convInfo.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${filterWidth}; wC++) {
              int xC = xCCorner + wC * ${dilationWidth};

              if (xC < 0 || xC >= ${convInfo.inWidth}) {
                continue;
              }

              for (int d1 = 0; d1 < ${inputDepthNearestVec4}; d1 += 4) {
                vec4 xValues = vec4(
                  getX(batch, xF, xR, xC, d1),
                  getX(batch, xF, xR, xC, d1 + 1),
                  getX(batch, xF, xR, xC, d1 + 2),
                  getX(batch, xF, xR, xC, d1 + 3)
                );
                vec4 wValues = vec4(
                  getW(wF, wR, wC, d1, d2),
                  getW(wF, wR, wC, d1 + 1, d2),
                  getW(wF, wR, wC, d1 + 2, d2),
                  getW(wF, wR, wC, d1 + 3, d2)
                );

                dotProd += dot(xValues, wValues);
              }

              if (${inputDepthVec4Remainder === 1}) {
                dotProd +=
                  getX(batch, xF, xR, xC, ${inputDepthNearestVec4}) *
                  getW(wF, wR, wC, ${inputDepthNearestVec4}, d2);
              } else if (${inputDepthVec4Remainder === 2}) {
                vec2 xValues = vec2(
                  getX(batch, xF, xR, xC, ${inputDepthNearestVec4}),
                  getX(batch, xF, xR, xC, ${inputDepthNearestVec4} + 1)
                );
                vec2 wValues = vec2(
                  getW(wF, wR, wC, ${inputDepthNearestVec4}, d2),
                  getW(wF, wR, wC, ${inputDepthNearestVec4} + 1, d2)
                );
                dotProd += dot(xValues, wValues);
              } else if (${inputDepthVec4Remainder === 3}) {
                vec3 xValues = vec3(
                  getX(batch, xF, xR, xC, ${inputDepthNearestVec4}),
                  getX(batch, xF, xR, xC, ${inputDepthNearestVec4} + 1),
                  getX(batch, xF, xR, xC, ${inputDepthNearestVec4} + 2)
                );
                vec3 wValues = vec3(
                  getW(wF, wR, wC, ${inputDepthNearestVec4}, d2),
                  getW(wF, wR, wC, ${inputDepthNearestVec4} + 1, d2),
                  getW(wF, wR, wC, ${inputDepthNearestVec4} + 2, d2)
                );
                dotProd += dot(xValues, wValues);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `;
  }
}
