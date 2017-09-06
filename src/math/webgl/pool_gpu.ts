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

export class Pool2DProgram implements GPGPUProgram {
  variableNames = ['x'];
  params: Array<{}>;
  outputShape: number[];
  userCode: string;

  constructor(
      xShape: [number, number, number], fSize: number, stride: number,
      pad: number, poolType: 'max'|'min'|'avg', computePositions: boolean) {
    if (poolType === 'avg' && computePositions) {
      throw new Error('Cannot compute positions for average pool.');
    }

    let returnValue = 'minMaxValue';
    if (computePositions) {
      returnValue = 'float(minMaxPosition)';
    } else if (poolType === 'avg') {
      returnValue = `avgValue / ${fSize * fSize}.0`;
    }
    const xRowsLimit = xShape[0];
    const xColsLimit = xShape[1];
    this.params = [stride, pad, fSize, poolType, computePositions];
    this.outputShape =
        conv_util.computeOutputShape3D(xShape, fSize, xShape[2], stride, pad);

    const isAvgPool = poolType === 'avg';
    const compareOp = poolType === 'min' ? '<=' : '>=';

    this.userCode = `
      const ivec2 strides = ivec2(${stride}, ${stride});
      const ivec2 pads = ivec2(${pad}, ${pad});

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

        for (int wR = 0; wR < ${fSize}; wR++) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${xRowsLimit}) {
            continue;
          }

          for (int wC = 0; wC < ${fSize}; wC++) {
            int xC = xCCorner + wC;

            if (xC < 0 || xC >= ${xColsLimit}) {
              continue;
            }

            float value = getX(xR, xC, d);

            if (isNaN(value)) {
              setOutput(value);
              return;
            }

            if (${isAvgPool}) {
              avgValue += value;
            } else {
              // If a min / max value has already been found, use it. If not,
              // use the current value.
              float currMinMaxValue = mix(
                  value, minMaxValue, minMaxValueFound);
              if (value ${compareOp} currMinMaxValue) {
                minMaxValue = value;
                minMaxValueFound = 1.0;
                if (${computePositions}) {
                  minMaxPosition = wR * ${fSize} + wC;
                }
              }
            }
          }
        }
        setOutput(${returnValue});
      }
    `;
  }
}
