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

import {GPGPUContext} from './gpgpu_context';
import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class SliceProgram implements GPGPUProgram {
  variableNames = ['source'];
  outputShape: number[];
  userCode: string;
  rank: number;

  // Caching uniform location for speed.
  startLoc: WebGLUniformLocation;

  constructor(destSize: number[]) {
    this.outputShape = destSize;
    this.rank = destSize.length;

    const dtype = getCoordsDataType(this.rank);
    const uniformPart = `uniform int start[${this.rank}];`;
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
      ${uniformPart}
      void main() {
        ${body}
        setOutput(getSource(${sourceCoords}));
      }
    `;
  }

  getCustomSetupFunc(start: number[]) {
    if (start.length !== this.rank) {
      throw Error(
          `The rank (${this.rank}) of the program must match the ` +
          `length of start (${start.length})`);
    }
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      if (this.startLoc == null) {
        this.startLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'start');
        if (this.startLoc == null) {
          // This means the compiler has optimized and realized it doesn't need
          // the uniform.
          return;
        }
      }
      gpgpu.gl.uniform1iv(this.startLoc, start);
    };
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
