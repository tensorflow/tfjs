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

function buildUserCode(innerCodes: string) {
  return `
   void main() {
    ivec4 coords = getOutputCoords();
    vec4 result = vec4(0.0);
    ${innerCodes}
    setOutput(result);
   }
 `;
}

export class SquareReadProgram implements GPGPUProgram {
  variableNames = ['x', 'W'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[];
  userCode: string;


  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.outShape;
    const innerCodes = `
    for (int row = 0; row < ${Math.ceil(convInfo.inHeight / 4)}; row += 4) {
          for (int col = 0; col < ${
        Math.ceil(convInfo.inWidth / 4)}; col += 4) {
              result = getX(0, 0, row, col);
              if (col + 2 < ${convInfo.inWidth}) {
                result = getX(0, 0, row, col + 2);
              }
              if (row + 2 < ${convInfo.inHeight}) {
                result = getX(0, 0, row + 2, col);
                if (col + 2 < ${convInfo.inWidth}) {
                  result = getX(0, 0, row + 2, col + 2);
                }
              }
          }
    }
    `;
    this.userCode = buildUserCode(innerCodes);
  }
}

export class LinearReadProgram implements GPGPUProgram {
  variableNames = ['x', 'W'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[];
  userCode: string;


  constructor(convInfo: backend_util.Conv2DInfo, direction = 1) {
    this.outputShape = convInfo.outShape;
    let innerCodes;
    if (direction == 1) {  // Horizontal read
      innerCodes = `
      for (int row = 0; row < ${Math.ceil(convInfo.inHeight / 2)}; row += 2) {
        for (int col = 0; col < ${Math.ceil(convInfo.inWidth / 8)}; col += 8) {
                result = getX(0, 0, row, col);
                if (col + 2 < ${convInfo.inWidth}) {
                  result = getX(0, 0, row, col + 2);
                }
                if (col + 4 < ${convInfo.inWidth}) {
                  result = getX(0, 0, row, col + 4);
                }
                if (col + 6 < ${convInfo.inWidth}) {
                  result = getX(0, 0, row, col + 6);
                }
            }
      }
      `;
    } else if (direction === 2) {  // Vertical read
      innerCodes = `
      for (int col = 0; col < ${Math.ceil(convInfo.inWidth / 2)}; col += 2) {
        for (int row = 0; row < ${Math.ceil(convInfo.inHeight / 8)}; row += 8) {
                result = getX(0, 0, row, col);
                if (row + 2 < ${convInfo.inHeight}) {
                  result = getX(0, 0, row + 2, col);
                }
                if (row + 4 < ${convInfo.inHeight}) {
                  result = getX(0, 0, row + 4, col);
                }
                if (row + 6 < ${convInfo.inHeight}) {
                  result = getX(0, 0, row + 6, col);
                }
            }
      }
      `;
    }
    this.userCode = buildUserCode(innerCodes);
  }
}
