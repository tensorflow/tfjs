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

import {ConvInfo} from '../conv_util';
import {GPGPUProgram} from './gpgpu_math';

export class Pool2DProgram implements GPGPUProgram {
  variableNames = ['x'];
  params: Array<{}>;
  outputShape: number[];
  userCode: string;

  constructor(
      convInfo: ConvInfo, poolType: 'max'|'min'|'avg',
      computePositions: boolean) {
    if (poolType === 'avg' && computePositions) {
      throw new Error('Cannot compute positions for average pool.');
    }

    const filterHeight = convInfo.filterHeight;
    const filterWidth = convInfo.filterWidth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;

    const xNumRows = convInfo.inShape[0];
    const xNumCols = convInfo.inShape[1];
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    this.params = [
      strideHeight, strideWidth, padLeft, padTop, poolType, computePositions
    ];
    this.outputShape = convInfo.outShape;

    const isAvgPool = poolType === 'avg';

    let initializationValue = '0.0';
    if (!isAvgPool) {
      if (poolType === 'min') {
        initializationValue = '1.0 / 0.0';
      } else {
        initializationValue = '-1.0 / 0.0';
      }
    }

    if (computePositions) {
      const compareOp = poolType === 'min' ? '<=' : '>=';

      this.userCode = `
        const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
        const ivec2 pads = ivec2(${padTop}, ${padLeft});

        void main() {
          ivec3 coords = getOutputCoords();
          int d = coords.z;

          ivec2 xRCCorner = coords.xy * strides - pads;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          // max/min x(?, ?, d) to get y(yR, yC, d).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;
          float avgValue = 0.0;

          for (int wR = 0; wR < ${filterHeight}; wR++) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${xNumRows}) {
              continue;
            }

            for (int wC = 0; wC < ${filterWidth}; wC++) {
              int xC = xCCorner + wC;

              if (xC < 0 || xC >= ${xNumCols}) {
                continue;
              }

              float value = getX(xR, xC, d);

              if (isNaN(value)) {
                setOutput(value);
                return;
              }

              // If a min / max value has already been found, use it. If not,
              // use the current value.
              float currMinMaxValue = mix(
                  value, minMaxValue, minMaxValueFound);
              if (value ${compareOp} currMinMaxValue) {
                minMaxValue = value;
                minMaxValueFound = 1.0;
                minMaxPosition = wR * ${filterWidth} + wC;
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;
      return;
    }

    const compareOp = poolType === 'min' ? 'min' : 'max';

    let returnValue = `${poolType}(${poolType}(${poolType}(` +
        'minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])';
    if (poolType === 'avg') {
      returnValue = `avgValue / ${filterHeight * filterWidth}.0`;
    }

    const filterWidthNearestVec4 = Math.floor(filterWidth / 4) * 4;
    const filterWidthVec4Remainder = filterWidth % 4;

    const updateSnippet = `
      if (hasNaN(values)) {
        setOutput(getNaN(values));
        return;
      }
      if (${isAvgPool}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${compareOp}(values, minMaxValue);
      }
    `;

    this.userCode = `
      const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
      const ivec2 pads = ivec2(${padTop}, ${padLeft});
      const float initializationValue = ${initializationValue};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int xR, int xC, int d) {
        if (xC < 0 || xC >= ${xNumCols}) {
          return initializationValue;
        }
        return getX(xR, xC, d);
      }

      void main() {
        ivec3 coords = getOutputCoords();
        int d = coords.z;

        ivec2 xRCCorner = coords.xy * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // max/min x(?, ?, d) to get y(yR, yC, d).
        // ? = to be determined
        vec4 minMaxValue = vec4(${initializationValue});
        float avgValue = 0.0;

        for (int wR = 0; wR < ${filterHeight}; wR++) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${xNumRows}) {
            continue;
          }

          for (int wC = 0; wC < ${filterWidthNearestVec4}; wC += 4) {
            int xC = xCCorner + wC;

            vec4 values = vec4(
              getValue(xR, xC, d),
              getValue(xR, xC + 1, d),
              getValue(xR, xC + 2, d),
              getValue(xR, xC + 3, d)
            );

            ${updateSnippet}
          }

          int xC = xCCorner + ${filterWidthNearestVec4};
          if (${filterWidthVec4Remainder === 1}) {
            vec4 values = vec4(
              getValue(xR, xC, d),
              initializationValue,
              initializationValue,
              initializationValue
            );
            ${updateSnippet}
          } else if (${filterWidthVec4Remainder === 2}) {
            vec4 values = vec4(
              getValue(xR, xC, d),
              getValue(xR, xC + 1, d),
              initializationValue,
              initializationValue
            );

            ${updateSnippet}
          } else if (${filterWidthVec4Remainder === 3}) {
            vec4 values = vec4(
              getValue(xR, xC, d),
              getValue(xR, xC + 1, d),
              getValue(xR, xC + 2, d),
              initializationValue
            );

            ${updateSnippet}
          }
        }
        setOutput(${returnValue});
      }
    `;
  }
}
