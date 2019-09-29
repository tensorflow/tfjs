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

import {ReduceInfo} from '../../ops/reduce_util';
import {GPGPUProgram} from './gpgpu_math';

export class ArgMinMaxProgram implements GPGPUProgram {
  variableNames = ['A'];
  outputShape: number[];
  userCode: string;

  constructor(reduceInfo: ReduceInfo, op: 'max'|'min', firstPass: boolean) {
    const windowSize = reduceInfo.windowSize;
    const batchSize = reduceInfo.batchSize;
    const inSize = reduceInfo.inSize;
    const outSize = Math.ceil(inSize / windowSize);
    if (!firstPass) {
      this.variableNames.push('bestIndicesA');
    }
    this.outputShape = [batchSize, outSize];
    const compOp = (op === 'max') ? '>' : '<';
    const indexSnippet = firstPass ?
        'inOffset + i' :
        'round(getBestIndicesA(batch, inOffset + i))';

    this.userCode = `
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${windowSize};
        int i = 0;
        int inIdx = ${indexSnippet};
        int bestIndex = inIdx;
        float bestValue = getA(batch, bestIndex);

        for (int i = 0; i < ${windowSize}; i++) {
          inIdx = ${indexSnippet};
          float candidate = getA(batch, inIdx);
          if (candidate ${compOp} bestValue) {
            bestValue = candidate;
            bestIndex = inIdx;
          }
        }
        setOutput(float(bestIndex));
      }
    `;
  }
}
