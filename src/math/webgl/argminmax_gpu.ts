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

export function getArgMinMaxSnippet(
    op: 'min'|'max', texName: string, size: number): string {
  const compOp = (op === 'min') ? '<' : '>';
  return `
    float getArgMinMax${texName}() {
      int bestIndex = 0;
      float bestValue = get${texName}Flat(0);

      for (int i = 0; i < ${size}; i++) {
        float candidate = get${texName}Flat(i);
        if (isNaN(candidate)) {
          return candidate;
        }
        if (candidate ${compOp} bestValue) {
          bestValue = candidate;
          bestIndex = i;
        }
      }
      return float(bestIndex);
    }
  `;
}

export class ArgMinMaxProgram implements GPGPUProgram {
  variableNames = ['A'];
  outputShape: number[] = [];
  params: Array<{}>;
  userCode: string;

  constructor(aSize: number, opType: 'min'|'max') {
    this.params = [opType];
    const aSnippet = getArgMinMaxSnippet(opType, 'A', aSize);
    this.userCode = `
      ${aSnippet}

      void main() {
        setOutput(getArgMinMaxA());
      }
    `;
  }
}
