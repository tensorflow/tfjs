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

import * as concat_util from '../../concat_util';
import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class ConcatProgram implements GPGPUProgram {
  variableNames = ['A', 'B'];
  outputShape: number[] = [];
  userCode: string;

  constructor(aShape: number[], bShape: number[], axis: number) {
    const yAxes = ['yR', 'yC', 'yD', 'yW'];
    const concatAxis = yAxes[axis];
    this.outputShape = concat_util.computeOutShape(aShape, bShape, axis);

    const dType = getCoordsDataType(aShape.length);
    const unpackSnippet = getUnpack(aShape.length);
    const sampleCoords = getSampleCoords(aShape.length);

    this.userCode = `
      void main() {
        ${dType} coords = getOutputCoords();
        ${unpackSnippet}

        float value = 0.0;
        if (${concatAxis} < ${aShape[axis]}) {
          value = getA(${sampleCoords});
        } else {
          ${concatAxis} -= ${aShape[axis]};
          value = getB(${sampleCoords});
        }

        setOutput(value);
      }
    `;
  }
}

function getSampleCoords(rank: number): string {
  if (rank === 1) {
    return 'yR';
  } else if (rank === 2) {
    return 'yR, yC';
  } else if (rank === 3) {
    return 'yR, yC, yD';
  } else if (rank === 4) {
    return 'yR, yC, yD, yW';
  } else {
    throw Error(`Concat for rank ${rank} is not yet supported`);
  }
}

function getUnpack(rank: number): string {
  let res = rank === 1 ? 'int yR = coords;' : 'int yR = coords.x;';
  if (rank > 1) {
    res += '\nint yC = coords.y;';
  }
  if (rank > 2) {
    res += '\nint yD = coords.z;';
  }
  if (rank > 3) {
    res += '\nint yW = coords.w;';
  }
  if (rank > 4) {
    throw Error(`Concat for rank ${rank} is not yet supported`);
  }
  return res;
}
