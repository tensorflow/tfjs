/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

export class CumSumProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  constructor(shape: number[], exclusive: boolean, reverse: boolean) {
    this.outputShape = shape;
    const rank = shape.length;
    const finalDim = shape[shape.length - 1];
    const comparator = reverse ? '<' : '>';

    this.userCode = `
      int getIndex(int i) {
        ${reverse ? `return ${finalDim} -i - 1;` : 'return i;'}
      }

      void main() {
        ${getCoordsDataType(rank)} coords = getOutputCoords();
        int end = ${getFinalCoord(rank, 'coords')};
        float val = 0.0;
        for (int i = ${finalDim} - 1; i >= 0; i -= 1) {
          int idx = getIndex(i);
          if (idx ${comparator} end) {
            continue;
          }
          if (idx == end && ${exclusive}) {
            continue;
          }
          ${getFinalCoord(rank, 'coords')} = idx;
          val += getX(${getCoords(rank, 'coords')});
        }
        setOutput(val);
      }
    `;
  }
}

function getCoords(rank: number, name: string): string {
  if (rank === 1) {
    return `${name}`;
  } else if (rank === 2) {
    return `${name}.x, ${name}.y`;
  } else if (rank === 3) {
    return `${name}.x, ${name}.y, ${name}.z`;
  } else if (rank === 4) {
    return `${name}.x, ${name}.y, ${name}.z, ${name}.w`;
  } else {
    throw Error(`Cumulative sum for rank ${rank} is not yet supported`);
  }
}

function getFinalCoord(rank: number, name: string): string {
  if (rank === 1) {
    return `${name}`;
  } else if (rank === 2) {
    return `${name}.y`;
  } else if (rank === 3) {
    return `${name}.z`;
  } else if (rank === 4) {
    return `${name}.w`;
  } else {
    throw Error(`Cumulative sum for rank ${rank} is not yet supported`);
  }
}
