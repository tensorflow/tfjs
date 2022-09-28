/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

export class Conv2DDensePackedProgram implements GPGPUProgram {
  variableNames = ['x', 'W'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[];
  userCode: string;

  constructor(
    convInfo: backend_util.Conv2DInfo, addBias = false,
    activation: string = null, hasPreluActivationWeights = false,
    hasLeakyreluAlpha = false) {
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
    } else if (hasLeakyreluAlpha) {
      activationSnippet = `float activation(float a) {
        float b = getLeakyreluAlphaAtOutCoords();
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

  if (hasLeakyreluAlpha) {
    this.variableNames.push('leakyreluAlpha');
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
      vec4 dotProd = vec4(0.000000000000001);
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
            vec4 wValues0 = getW(wR, wC, d1, d2);
            vec4 wValues1 = getW(wR, wC, d1 + 1, d2);
            vec4 wValues2 = getW(wR, wC, d1 + 2, d2);
            vec4 wValues3 = getW(wR, wC, d1 + 3, d2);

            if (${isChannelsLast}) {
              vec4 xValues = getX(batch, xR, xC, d1);
              dotProd[0] += dot(xValues, vec4(wValues0.x, wValues1.x, wValues2.x, wValues3.x));
              dotProd[1] += dot(xValues, vec4(wValues0.y, wValues1.y, wValues2.y, wValues3.y));
              dotProd[2] += dot(xValues, vec4(wValues0.z, wValues1.z, wValues2.z, wValues3.z));
              dotProd[3] += dot(xValues, vec4(wValues0.w, wValues1.w, wValues2.w, wValues3.w));
            } else {
            }
          }

          // Assume both inputChannels and outputChannels are multiples of 4.
          if (${inputDepthVec4Remainder === 1}) {
          } else if (${inputDepthVec4Remainder === 2}) {
          } else if (${inputDepthVec4Remainder === 3}) {
          }
        }
      }

      vec4 result = dotProd - vec4(0.000000000000001);
      ${addBiasSnippet}
      ${applyActivationSnippet}
      setOutput(result);
    }
  `;
}
}
