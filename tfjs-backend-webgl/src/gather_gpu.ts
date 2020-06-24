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

import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class GatherProgram implements GPGPUProgram {
  variableNames = ['A', 'indices'];
  outputShape: number[];
  userCode: string;
  rank: number;

  constructor(aShape: number[], indicesLength: number, axis: number) {
    const outputShape: number[] = aShape.slice();
    outputShape[axis] = indicesLength;
    this.outputShape = outputShape;
    this.rank = outputShape.length;
    const dtype = getCoordsDataType(this.rank);
    const sourceCoords = getSourceCoords(aShape, axis);

    this.userCode = `
      void main() {
        ${dtype} resRC = getOutputCoords();
        setOutput(getA(${sourceCoords}));
      }
    `;
  }
}

function getSourceCoords(aShape: number[], axis: number): string {
  const rank = aShape.length;
  if (rank > 4) {
    throw Error(`Gather for rank ${rank} is not yet supported`);
  }
  if (rank === 1) {
    return `int(getIndices(resRC))`;
  }

  const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w'];

  const sourceCoords = [];
  for (let i = 0; i < aShape.length; i++) {
    if (i === axis) {
      sourceCoords.push(`int(getIndices(${currentCoords[i]}))`);
    } else {
      sourceCoords.push(`${currentCoords[i]}`);
    }
  }
  return sourceCoords.join();
}
