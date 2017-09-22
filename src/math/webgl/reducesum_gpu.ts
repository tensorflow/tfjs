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

export class ReduceSumProgram implements GPGPUProgram {
  variableNames = ['A'];
  params: Array<{}> = [];
  outputShape: number[] = [];
  userCode: string;

  constructor(public size: number) {
    const sizeNearestVec4 = Math.floor(size / 4) * 4;
    const sizeVec4Remainder = size % 4;

    const r1 = sizeNearestVec4;
    const r2 = sizeNearestVec4 + 1;
    const r3 = sizeNearestVec4 + 2;

    this.userCode = `
      void main() {
        const vec2 ones2 = vec2(1);
        const vec3 ones3 = vec3(1);
        const vec4 ones4 = vec4(1);

        float sum = 0.0;
        for (int i = 0; i < ${sizeNearestVec4}; i += 4) {
          vec4 aVec = vec4(getAFlat(i), getAFlat(i+1),
                           getAFlat(i+2), getAFlat(i+3));
          sum += dot(ones4, aVec);
        }

        if (${sizeVec4Remainder === 1}) {
          sum += getAFlat(${r1});
        } else if (${sizeVec4Remainder === 2}) {
          vec2 aVec = vec2(getAFlat(${r1}), getAFlat(${r2}));
          sum += dot(ones2, aVec);
        } else if (${sizeVec4Remainder === 3}) {
          vec3 aVec = vec3(getAFlat(${r1}), getAFlat(${r2}), getAFlat(${r3}));
          sum += dot(ones3, aVec);
        }

        setOutput(sum);
      }
    `;
  }
}
