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

import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class SelectProgram implements GPGPUProgram {
  variableNames = ['c', 'a', 'b'];
  outputShape: number[];
  userCode: string;

  constructor(cRank: number, shape: number[], rank: number) {
    this.outputShape = shape;

    let cCoords;
    let abCoords;
    if (rank > 4) {
      throw Error(`Where for rank ${rank} is not yet supported`);
    }

    if (rank === 1) {
      abCoords = `resRC`;
      cCoords = `resRC`;
    } else {
      const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];
      const cCoordVars = [];
      const abCoordVars = [];
      for (let i = 0; i < shape.length; i++) {
        abCoordVars.push(`${currentCoords[i]}`);
        if (i < cRank) {
          cCoordVars.push(`${currentCoords[i]}`);
        }
      }
      cCoords = cCoordVars.join();
      abCoords = abCoordVars.join();
    }

    const dtype = getCoordsDataType(rank);

    this.userCode = `
      void main() {
        ${dtype} resRC = getOutputCoords();
        float cVal = getC(${cCoords});
        if (cVal >= 1.0) {
          setOutput(getA(${abCoords}));
        } else {
          setOutput(getB(${abCoords}));
        }
      }
    `;
  }
}
