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

import {GPGPUProgram} from './gpgpu_math';

export class MinMaxProgram implements GPGPUProgram {
  variableNames = ['A'];
  params: Array<{}>;
  outputShape: number[] = [];
  userCode: string;

  constructor(size: number, op: 'min'|'max') {
    this.params = [op];
    const sizeNearestVec4 = Math.floor(size / 4) * 4;
    const sizeVec4Remainder = size % 4;

    const r1 = sizeNearestVec4;
    const r2 = sizeNearestVec4 + 1;
    const r3 = sizeNearestVec4 + 2;

    this.userCode = `
      void main() {
        vec4 bestVec = vec4(getAFlat(0));
        for (int i = 0; i < ${sizeNearestVec4}; i += 4) {
          vec4 aVec = vec4(getAFlat(i), getAFlat(i+1),
                           getAFlat(i+2), getAFlat(i+3));
          if (hasNaN(aVec)) {
            setOutput(getNaN(aVec));
            return;
          }
          bestVec = ${op}(bestVec, aVec);
        }
        vec4 aVec;
        if (${sizeVec4Remainder === 1}) {
          aVec = vec4(bestVec.xyz, getAFlat(${r1}));
        } else if (${sizeVec4Remainder === 2}) {
          aVec = vec4(bestVec.xy, vec2(getAFlat(${r1}), getAFlat(${r2})));
        } else if (${sizeVec4Remainder === 3}) {
          aVec = vec4(bestVec.x,
                      vec3(getAFlat(${r1}), getAFlat(${r2}), getAFlat(${r3})));
        }
        if (${sizeVec4Remainder > 0}) {
          if (hasNaN(aVec)) {
            setOutput(getNaN(aVec));
            return;
          }
          bestVec = ${op}(bestVec, aVec);
        }

        float final = ${op}(bestVec.x, ${op}(bestVec.y,
                      ${op}(bestVec.z, bestVec.w)));
        setOutput(final);
      }
    `;
  }
}
