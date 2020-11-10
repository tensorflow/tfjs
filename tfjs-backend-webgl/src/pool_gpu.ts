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

export class Pool2DProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  constructor(
      convInfo: backend_util.Conv2DInfo, poolType: 'max'|'avg',
      computePositions: boolean, flattenPositions = false,
      includeBatchInIndex = false) {
    if (poolType === 'avg' && computePositions) {
      throw new Error('Cannot compute positions for average pool.');
    }

    const filterWidth = convInfo.filterWidth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;

    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    this.outputShape = convInfo.outShape;

    const isAvgPool = poolType === 'avg';
    const batchFlattenPositionStr = `((batch  * ${convInfo.inHeight} + xR) * ${
        convInfo.inWidth} + xC) * ${convInfo.inChannels} + d`;
    const flattenPositionStr =
        `(xR * ${convInfo.inWidth} + xC) * ${convInfo.inChannels} + d`;

    let initializationValue = '0.0';
    if (!isAvgPool) {
      // WebGL on Firefox Linux can't compile 1/0 so we do 1/eps.
      initializationValue = '-1.0 / 1e-20';
    }

    if (computePositions) {
      const compareOp = '>=';

      this.userCode = `
        const ivec2 strides = ivec2(${strideHeight}, ${strideWidth});
        const ivec2 pads = ivec2(${padTop}, ${padLeft});

        void main() {
          ivec4 coords = getOutputCoords();
          int batch = coords[0];
          int d = coords[3];

          ivec2 xRCCorner = coords.yz * strides - pads;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          // max/min x(?, ?, d) to get y(yR, yC, d).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;
          float avgValue = 0.0;

          for (int wR = 0; wR < ${effectiveFilterHeight};
              wR += ${dilationHeight}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${convInfo.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${effectiveFilterWidth};
                wC += ${dilationWidth}) {
              int xC = xCCorner + wC;

              if (xC < 0 || xC >= ${convInfo.inWidth}) {
                continue;
              }

              float value = getX(batch, xR, xC, d);

              // If a min / max value has already been found, use it. If not,
              // use the current value.
              float currMinMaxValue = mix(
                  value, minMaxValue, minMaxValueFound);
              if (value ${compareOp} currMinMaxValue) {
                minMaxValue = value;
                minMaxValueFound = 1.0;
                minMaxPosition = ${
          flattenPositions ? (includeBatchInIndex ? batchFlattenPositionStr :
                                                    flattenPositionStr) :
                             `wR * ${effectiveFilterWidth} + wC`};
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;
      return;
    }

    const compareOp = 'max';

    let returnValue = `${poolType}(${poolType}(${poolType}(` +
        'minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])';
    if (poolType === 'avg') {
      returnValue = `avgValue / count`;
    }

    const filterWidthNearestVec4 = Math.floor(filterWidth / 4) * 4;
    const filterWidthVec4Remainder = filterWidth % 4;

    const updateSnippet = `
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

      float count = 0.0;

      float getValue(int batch, int xR, int xC, int d) {
        if (xC < 0 || xC >= ${convInfo.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xR, xC, d);
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // max/min x(?, ?, d) to get y(yR, yC, d).
        // ? = to be determined
        vec4 minMaxValue = vec4(${initializationValue});
        float avgValue = 0.0;
        count = 0.0;

        for (int wR = 0; wR < ${effectiveFilterHeight};
            wR += ${dilationHeight}) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${convInfo.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${filterWidthNearestVec4}; wC += 4) {
            int xC = xCCorner + wC * ${dilationWidth};

            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${dilationWidth}, d),
              getValue(batch, xR, xC + 2 * ${dilationWidth}, d),
              getValue(batch, xR, xC + 3 * ${dilationWidth}, d)
            );

            ${updateSnippet}
          }

          int xC = xCCorner + ${filterWidthNearestVec4};
          if (${filterWidthVec4Remainder === 1}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              initializationValue,
              initializationValue,
              initializationValue
            );

            ${updateSnippet}
          } else if (${filterWidthVec4Remainder === 2}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${dilationWidth}, d),
              initializationValue,
              initializationValue
            );

            ${updateSnippet}
          } else if (${filterWidthVec4Remainder === 3}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${dilationWidth}, d),
              getValue(batch, xR, xC + 2 * ${dilationWidth}, d),
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

export class Pool3DProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  constructor(
      convInfo: backend_util.Conv3DInfo, poolType: 'max'|'avg',
      computePositions: boolean, flattenPositions = false,
      includeBatchInIndex = false) {
    if (poolType === 'avg' && computePositions) {
      throw new Error('Cannot compute positions for average pool.');
    }

    const filterWidth = convInfo.filterWidth;
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;

    const padFront = convInfo.padInfo.front;
    const padTop = convInfo.padInfo.top;
    const padLeft = convInfo.padInfo.left;
    this.outputShape = convInfo.outShape;

    const isAvgPool = poolType === 'avg';

    let initializationValue = '0.0';
    if (!isAvgPool) {
      // WebGL on Firefox Linux can't compile 1/0 so we do 1/eps.
      initializationValue = '-1.0 / 1e-20';
    }

    if (computePositions) {
      const compareOp = '>=';

      this.userCode = `
        const ivec3 strides =
            ivec3(${strideDepth}, ${strideHeight}, ${strideWidth});
        const ivec3 pads = ivec3(${padFront}, ${padTop}, ${padLeft});

        void main() {
          ivec5 coords = getOutputCoords();
          int batch = coords.x;
          int ch = coords.u;

          ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
          int xDCorner = xCorner.x;
          int xRCorner = xCorner.y;
          int xCCorner = xCorner.z;

          // max/min x(?, ?, ?, ch) to get y(yD, yR, yC, ch).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;

          for (int wD = 0; wD < ${effectiveFilterDepth};
              wD += ${dilationDepth}) {
            int xD = xDCorner + wD;

            if (xD < 0 || xD >= ${convInfo.inDepth}) {
              continue;
            }

            for (int wR = 0; wR < ${effectiveFilterHeight};
                wR += ${dilationHeight}) {
              int xR = xRCorner + wR;

              if (xR < 0 || xR >= ${convInfo.inHeight}) {
                continue;
              }

              for (int wC = 0; wC < ${effectiveFilterWidth};
                  wC += ${dilationWidth}) {
                int xC = xCCorner + wC;

                if (xC < 0 || xC >= ${convInfo.inWidth}) {
                  continue;
                }

                float value = getX(batch, xD, xR, xC, ch);

                // If a min / max value has already been found, use it. If not,
                // use the current value.
                float currMinMaxValue = mix(
                    value, minMaxValue, minMaxValueFound);
                if (value ${compareOp} currMinMaxValue) {
                  minMaxValue = value;
                  minMaxValueFound = 1.0;
                  minMaxPosition = ${
          flattenPositions ?
              (includeBatchInIndex ?
                   `(((batch * ${convInfo.inDepth} + xD) * ${
                       convInfo.inHeight} + xR) * ${convInfo.inWidth} + xC) * ${
                       convInfo.inChannels} + ch` :
                   `((xD * ${convInfo.inHeight} + xR) * ${
                       convInfo.inWidth} + xC) * ${convInfo.inChannels} + ch`) :
              `wD * ${effectiveFilterHeight} * ${effectiveFilterWidth} +
                      wR * ${effectiveFilterWidth} + wC`};
                }
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;
      return;
    }

    const compareOp = 'max';

    let returnValue = `${poolType}(${poolType}(${poolType}(` +
        'minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])';
    if (poolType === 'avg') {
      returnValue = `avgValue / count`;
    }

    const filterWidthNearestVec4 = Math.floor(filterWidth / 4) * 4;
    const filterWidthVec4Remainder = filterWidth % 4;

    const updateSnippet = `
      if (${isAvgPool}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${compareOp}(values, minMaxValue);
      }
    `;

    this.userCode = `
      const ivec3 strides =
        ivec3(${strideDepth}, ${strideHeight}, ${strideWidth});
      const ivec3 pads = ivec3(${padFront}, ${padTop}, ${padLeft});
      const float initializationValue = ${initializationValue};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xD, int xR, int xC, int ch) {
        if (xC < 0 || xC >= ${convInfo.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xD, xR, xC, ch);
      }

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xDCorner = xCorner.x;
        int xRCorner = xCorner.y;
        int xCCorner = xCorner.z;

        // max/min x(?, ?, ?, d) to get y(yD, yR, yC, ch).
        // ? = to be determined
        vec4 minMaxValue = vec4(${initializationValue});
        float avgValue = 0.0;
        count = 0.0;

        for (int wD = 0; wD < ${effectiveFilterDepth};
            wD += ${dilationDepth}) {
          int xD = xDCorner + wD;

          if (xD < 0 || xD >= ${convInfo.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${effectiveFilterHeight};
            wR += ${dilationHeight}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${convInfo.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${filterWidthNearestVec4}; wC += 4) {
              int xC = xCCorner + wC * ${dilationWidth};

              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${dilationWidth}, ch),
                getValue(batch, xD, xR, xC + 2 * ${dilationWidth}, ch),
                getValue(batch, xD, xR, xC + 3 * ${dilationWidth}, ch)
              );

              ${updateSnippet}
            }

            int xC = xCCorner + ${filterWidthNearestVec4};
            if (${filterWidthVec4Remainder === 1}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                initializationValue,
                initializationValue,
                initializationValue
              );

              ${updateSnippet}
            } else if (${filterWidthVec4Remainder === 2}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${dilationWidth}, ch),
                initializationValue,
                initializationValue
              );

              ${updateSnippet}
            } else if (${filterWidthVec4Remainder === 3}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${dilationWidth}, ch),
                getValue(batch, xD, xR, xC + 2 * ${dilationWidth}, ch),
                initializationValue
              );

              ${updateSnippet}
            }
          }
          setOutput(${returnValue});
        }
      }
    `;
  }
}
