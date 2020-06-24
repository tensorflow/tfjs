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

export class TransposeProgram implements GPGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  userCode: string;
  rank: number;

  constructor(aShape: number[], newDim: number[]) {
    const outputShape: number[] = new Array(aShape.length);
    for (let i = 0; i < outputShape.length; i++) {
      outputShape[i] = aShape[newDim[i]];
    }
    this.outputShape = outputShape;
    this.rank = outputShape.length;
    const dtype = getCoordsDataType(this.rank);
    const switched = getSwitchedCoords(newDim);

    this.userCode = `
    void main() {
      ${dtype} resRC = getOutputCoords();
      setOutput(getA(${switched}));
    }
    `;
  }
}

function getSwitchedCoords(newDim: number[]): string {
  const rank = newDim.length;
  if (rank > 6) {
    throw Error(`Transpose for rank ${rank} is not yet supported`);
  }
  const originalOrder =
      ['resRC.x', 'resRC.y', 'resRC.z', 'resRC.w', 'resRC.u', 'resRC.v'];
  const switchedCoords = new Array(rank);
  for (let i = 0; i < newDim.length; i++) {
    switchedCoords[newDim[i]] = originalOrder[i];
  }
  return switchedCoords.join();
}
