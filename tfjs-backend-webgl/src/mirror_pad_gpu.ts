/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

export class MirrorPadProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  constructor(
      xShape: number[], paddings: Array<[number, number]>,
      mode: 'reflect'|'symmetric') {
    this.outputShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
    const rank = xShape.length;
    const dtype = getCoordsDataType(rank);

    const start = paddings.map(p => p[0]).join(',');
    const end = paddings.map((p, i) => p[0] + xShape[i]).join(',');
    const unpackedCoords =
        ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank);
    const offset = mode === 'reflect' ? 0 : 1;

    if (rank === 1) {
      this.userCode = `
        int start = ${start};
        int end = ${end};

        void main() {
          int outC = getOutputCoords();
          if (outC < start) {
            outC = start * 2 - outC - ${offset};
          } else if(outC >= end) {
            outC = (end - 1) * 2 - outC + ${offset};
          }
          setOutput(getX(outC - start));
        }
      `;
      return;
    }
    this.userCode = `
      ${dtype} start = ${dtype}(${start});
      ${dtype} end = ${dtype}(${end});

      void main() {
        ${dtype} outC = getOutputCoords();
        for (int i = 0; i < ${rank}; i++) {
          if (outC[i] < start[i]) {
            outC[i] = start[i] * 2 - outC[i] - ${offset};
          } else if(outC[i] >= end[i]) {
            outC[i] = (end[i] - 1) * 2 - outC[i] + ${offset};
          }
        }
        ${dtype} coords = outC - start;
        setOutput(getX(${unpackedCoords}));
      }
    `;
  }
}
