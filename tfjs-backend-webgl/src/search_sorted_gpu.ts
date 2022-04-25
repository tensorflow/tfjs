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

import {GPGPUProgram} from './gpgpu_math';
import {UniformType} from './shader_compiler';

export class SearchSortedProgram implements GPGPUProgram {
  variableNames = ['sortedSequence', 'values'];
  outputShape: number[];
  userCode: string;
  customUniforms = [{name: 'numInputs', type: 'int' as UniformType}];

  constructor(batchSize: number, numValues: number, side: 'left'|'right') {
    this.outputShape = [batchSize, numValues];

    const searchFunction = side === 'left' ? 'lowerBound' : 'upperBound';
    this.userCode = `
       int lowerBound(int batch, float value) {
         int left = 0;
         int right = numInputs;
         int mid;
         while (left < right) {
           mid = (left + right) / 2;
           if (getSortedSequence(batch, mid) < value) {
             left = mid + 1;
           } else {
             right = mid;
           }
         }
         return right;
       }

       int upperBound(int batch, float value) {
        int left = 0;
        int right = numInputs;
        int mid;
        while (left < right) {
           mid = (left + right) / 2;
           if (getSortedSequence(batch, mid) <= value) {
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

         setOutput(float(${searchFunction}(batch, value)));
       }
     `;
  }
}
