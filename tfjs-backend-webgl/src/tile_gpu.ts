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

export class TileProgram implements GPGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  userCode: string;
  rank: number;

  constructor(aShape: number[], reps: number[]) {
    const outputShape: number[] = new Array(aShape.length);
    for (let i = 0; i < outputShape.length; i++) {
      outputShape[i] = aShape[i] * reps[i];
    }
    this.outputShape = outputShape;
    this.rank = outputShape.length;
    const dtype = getCoordsDataType(this.rank);
    const sourceCoords = getSourceCoords(aShape);

    this.userCode = `
      void main() {
        ${dtype} resRC = getOutputCoords();
        setOutput(getA(${sourceCoords}));
      }
    `;
  }
}

function getSourceCoords(aShape: number[]): string {
  const rank = aShape.length;
  if (rank > 5) {
    throw Error(`Tile for rank ${rank} is not yet supported`);
  }
  if (rank === 1) {
    return `imod(resRC, ${aShape[0]})`;
  }

  const currentCoords = ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w', 'resRC.u'];

  const sourceCoords = [];
  for (let i = 0; i < aShape.length; i++) {
    sourceCoords.push(`imod(${currentCoords[i]}, ${aShape[i]})`);
  }
  return sourceCoords.join();
}
