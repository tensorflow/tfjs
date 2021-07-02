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
import {getCoordsDataType, UniformType} from './shader_compiler';

export class SliceProgram implements GPGPUProgram {
  variableNames = ['source'];
  outputShape: number[];
  userCode: string;
  rank: number;
  customUniforms: Array<{name: string; arrayIndex: number; type: UniformType;}>;

  constructor(destSize: number[]) {
    this.outputShape = destSize;
    this.rank = destSize.length;

    const dtype = getCoordsDataType(this.rank);
    this.customUniforms = [{name: 'start', arrayIndex: this.rank, type: 'int'}];
    const sourceCoords = getCoords(this.rank);

    let body: string;
    const coordSum = destSize.map((_, i) => {
      return `sourceLoc.${coords[i]} = start[${i}] + coords.${coords[i]};`;
    });
    body = `
        ${dtype} sourceLoc;
        ${dtype} coords = getOutputCoords();
        ${coordSum.join('\n')}
      `;
    this.userCode = `
      void main() {
        ${body}
        setOutput(getSource(${sourceCoords}));
      }
    `;
  }
}

const coords = ['x', 'y', 'z', 'w', 'u', 'v'];

function getCoords(rank: number): string {
  if (rank === 1) {
    return 'sourceLoc';
  } else if (rank <= 6) {
    return coords.slice(0, rank).map(x => 'sourceLoc.' + x).join(',');
  } else {
    throw Error(`Slicing for rank ${rank} is not yet supported`);
  }
}
