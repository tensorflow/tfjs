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

import {GPGPUContext} from './gpgpu_context';
import {GPGPUProgram} from './gpgpu_math';
import {getCoordsDataType} from './shader_compiler';

export class PadProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;
  valueLoc: WebGLUniformLocation;

  constructor(
      xShape: number[], paddings: Array<[number, number]>,
      constantValue: number) {
    this.outputShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
    const rank = xShape.length;
    const type = getCoordsDataType(rank);

    const start = paddings.map(p => p[0]).join(',');
    const end = paddings.map((p, i) => p[0] + xShape[i]).join(',');
    const unpackedCoords =
        ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank);

    if (rank === 1) {
      this.userCode = `
        int start = ${start};
        int end = ${end};
        uniform float value;

        void main() {
          int outC = getOutputCoords();
          if (outC < start || outC >= end) {
            setOutput(value);
          } else {
            setOutput(getX(outC - start));
          }
        }
      `;
      return;
    }
    this.userCode = `
      ${type} start = ${type}(${start});
      ${type} end = ${type}(${end});
      uniform float value;

      void main() {
        ${type} outC = getOutputCoords();
        if (any(lessThan(outC, start)) || any(greaterThanEqual(outC, end))) {
          setOutput(value);
        } else {
          ${type} coords = outC - start;
          setOutput(getX(${unpackedCoords}));
        }
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
