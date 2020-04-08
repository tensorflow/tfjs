/**
 * @license
 * Copyright 2019 Google Inc. All Rights Reserved.
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
import {getChannels} from './packing_util';
import {getCoordsDataType} from './shader_compiler';

export class SlicePackedProgram implements GPGPUProgram {
  variableNames = ['source'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[];
  userCode: string;
  rank: number;

  // Caching uniform location for speed.
  startLoc: WebGLUniformLocation;

  constructor(destSize: number[]) {
    this.outputShape = destSize;
    this.rank = destSize.length;

    const dtype = getCoordsDataType(this.rank);
    const coords = getChannels('coords', this.rank);
    const sourceLoc = getChannels('sourceLoc', this.rank);

    const innerDims =
        this.rank === 1 ? 'sourceLoc' : `vec2(${sourceLoc.slice(-2).join()})`;
    const getChannel =
        `getChannel(getSource(${sourceLoc.join()}), ${innerDims})`;
    const upperRow = `
      result.x = ${getChannel};
      if (++${coords[this.rank - 1]} < ${destSize[this.rank - 1]}) {
        ++${sourceLoc[this.rank - 1]};
        result.y = ${getChannel};
        --${sourceLoc[this.rank - 1]};
      }
    `;
    const lowerRow = this.rank === 1 ? '' : `
      --${coords[this.rank - 1]};
      if (++${coords[this.rank - 2]} < ${destSize[this.rank - 2]}) {
        ++${sourceLoc[this.rank - 2]};
        result.z = ${getChannel};
        if (++${coords[this.rank - 1]} < ${destSize[this.rank - 1]}) {
          ++${sourceLoc[this.rank - 1]};
          result.w = ${getChannel};
        }
      }
    `;

    const sourceLocSetup = this.rank <= 4 ?
        `sourceLoc = coords +
            ${dtype}(${destSize.map((_, i) => `start[${i}]`).join()});` :
        destSize.map((_, i) => `${sourceLoc[i]} = ${coords[i]} + start[${i}];`)
            .join('\n');
    this.userCode = `
      uniform int start[${this.rank}];
      void main() {
        ${dtype} coords = getOutputCoords();
        ${dtype} sourceLoc;
        ${sourceLocSetup}
        vec4 result = vec4(0.);
        ${upperRow}
        ${lowerRow}
        setOutput(result);
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
