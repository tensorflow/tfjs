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

export class ReverseProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  constructor(xShape: number[], axis: number[]) {
    const rank = xShape.length;
    if (rank > 4) {
      throw new Error(
          `WebGL backend: Reverse of rank-${rank} tensor is not yet supported`);
    }
    this.outputShape = xShape;

    if (rank === 1) {
      this.userCode = `
        void main() {
          int coord = getOutputCoords();
          setOutput(getX(${xShape[0]} - coord - 1));
        }
      `;
      return;
    }
    const getInCoord = (i: number) => {
      if (axis.indexOf(i) !== -1 && xShape[i] !== 1) {
        return `${xShape[i]} - coords[${i}] - 1`;
      }
      return `coords[${i}]`;
    };
    const inCoords = xShape.map((_, i) => getInCoord(i)).join(',');
    const type = getCoordsDataType(rank);

    this.userCode = `
      void main() {
        ${type} coords = getOutputCoords();
        setOutput(getX(${inCoords}));
      }
    `;
  }
}
