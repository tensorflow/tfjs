/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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

import {env} from '@tensorflow/tfjs-core';
import {GPGPUProgram} from './gpgpu_math';
import {UniformType} from './shader_compiler';

export class SearchSortedProgram implements GPGPUProgram {
  variableNames = ['sortedSequence', 'values'];
  outputShape: number[];
  userCode: string;
  customUniforms = [{name: 'numInputs', type: 'int' as UniformType}];

  constructor(
      batchSize: number, numInputs: number, numValues: number,
      side: 'left'|'right') {
    this.outputShape = [batchSize, numValues];

    const webGL2LoopHead = 'while (left < right) {';
    // WebGL1 doesn't accept non constant loop conditions, so upper bound loop
    // iterations.
    const webGL1LoopHead = `for (int i = 0; i < ${
        Math.ceil(Math.log2(numInputs + 1))}; ++i) { if (left >= right) break;`;
    const loopHead = env().getNumber('WEBGL_VERSION') === 2 ? webGL2LoopHead :
                                                              webGL1LoopHead;

    // left corresponds to lower bound and right to upper bound.
    const boundComparator = side === 'left' ? '<' : '<=';
    this.userCode = `
       int findBound(int batch, float value) {
         int left = 0;
         int right = numInputs;
         int mid;
         ${loopHead}
           mid = (left + right) / 2;
           if (getSortedSequence(batch, mid) ${boundComparator} value) {
             left = mid + 1;
           } else {
             right = mid;
           }
         }
         return right;
       }

       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int valueIndex = coords[1];

         float value = getValues(batch, valueIndex);

         setOutput(float(findBound(batch, value)));
       }
     `;
  }
}
