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
import {GPGPUProgram, useShapeUniforms} from './gpgpu_math';

export class DepthwiseConv2DProgram implements GPGPUProgram {
  variableNames = ['x', 'W'];
  outputShape: number[];
  userCode: string;
  enableShapeUniforms: boolean;
  customUniforms = [
    {name: 'pads', type: 'ivec2' as const },
    {name: 'strides', type: 'ivec2' as const },
    {name: 'dilations', type: 'ivec2' as const },
    {name: 'inDims', type: 'ivec2' as const },
  ];

  constructor(
      convInfo: backend_util.Conv2DInfo, addBias = false,
      activation: string = null, hasPreluActivation = false,
      hasLeakyReluAlpha = false) {
    this.outputShape = convInfo.outShape;
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);

    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const channelMul = convInfo.outChannels / convInfo.inChannels;

    let activationSnippet = '', applyActivationSnippet = '';
    if (activation) {
      if (hasPreluActivation) {
        activationSnippet = `float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${activation}
        }`;
      } else if (hasLeakyReluAlpha) {
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

    if (hasPreluActivation) {
      this.variableNames.push('preluActivationWeights');
    }
    if (hasLeakyReluAlpha) {
      this.variableNames.push('leakyreluAlpha');
    }

    this.userCode = `
      ${activationSnippet}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${channelMul};
        int q = d2 - d1 * ${channelMul};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        // TO DO(dsmilkov): Flatten the two for loops and vec4 the operations.
        for (int wR = 0; wR < ${filterHeight}; wR++) {
          int xR = xRCorner + wR * dilations[0];

          if (xR < 0 || xR >= inDims[0]) {
            continue;
          }

          for (int wC = 0; wC < ${filterWidth}; wC++) {
            int xC = xCCorner + wC * dilations[1];

            if (xC < 0 || xC >= inDims[1]) {
              continue;
            }

            float xVal = getX(batch, xR, xC, d1);
            float wVal = getW(wR, wC, d1, q);
            dotProd += xVal * wVal;
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
