/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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

export class PadPackedProgram implements GPGPUProgram {
  variableNames = ['x'];
  packedInputs = true;
  packedOutput = true;
  outputShape: number[];
  userCode: string;
  valueLoc: WebGLUniformLocation;

  constructor(
      xShape: number[], paddings: Array<[number, number]>,
      constantValue: number) {
    this.outputShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
    const rank = xShape.length;
    const dtype = getCoordsDataType(rank);

    const start = paddings.map(p => p[0]).join(',');
    const end = paddings.map((p, i) => p[0] + xShape[i]).join(',');
    const coords = getChannels('rc', rank);
    const source = getChannels('source', rank);
    const cLimit = `${coords[rank - 1]} < ${this.outputShape[rank - 1]}`;
    const innerDims =
        rank === 1 ? 'source' : `vec2(${source.slice(-2).join()})`;

    const componentSetup = [
      `${dtype} rc = outputLoc;`, `${coords[rank - 1]} += 1;
       if(${cLimit}) {
      `,
      rank === 1 ? '' : `}
       rc = outputLoc;
       ${coords[rank - 2]} += 1;
       if(${coords[rank - 2]} < ${this.outputShape[rank - 2]}) {`,
      rank === 1 ? '' : `  ${coords[rank - 1]} += 1;
         if(${cLimit}) {`
    ];

    const paddingArea = rank === 1 ?
        'rc < start || rc >= end' :
        'any(lessThan(rc, start)) || any(greaterThanEqual(rc, end))';
    let mainLoop = '';
    for (let i = 0, j = rank === 1 ? 2 : 4; i < j; i++) {
      mainLoop += `
        ${componentSetup[i]}
        if (${paddingArea}) {
          result[${i}] = float(value);
        } else {
          ${dtype} source = rc - start;
          result[${i}] = getChannel(getX(${source.join()}), ${innerDims});
        }
      `;
    }
    mainLoop += (rank === 1 ? `} ` : `}}`);

    this.userCode = `
      const ${dtype} start = ${dtype}(${start});
      const ${dtype} end = ${dtype}(${end});
      uniform float value;

      void main() {
        ${dtype} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${mainLoop}
        setOutput(result);
      }
    `;
  }

  getCustomSetupFunc(value: number) {
    return (gpgpu: GPGPUContext, webGLProgram: WebGLProgram) => {
      if (this.valueLoc == null) {
        this.valueLoc = gpgpu.getUniformLocationNoThrow(webGLProgram, 'value');
      }
      gpgpu.gl.uniform1f(this.valueLoc, value);
    };
  }
}
