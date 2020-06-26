/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

export class ScatterProgram implements GPGPUProgram {
  variableNames = ['updates', 'indices', 'defaultValue'];
  outputShape: number[];
  userCode: string;

  constructor(
      updateSize: number, sliceDim: number, indicesRank: number,
      updatesRank: number, strides: number[], shape: number[],
      summingDupeIndex = true) {
    this.outputShape = shape;
    const stridesType = getCoordsDataType(strides.length);
    const dtype = getCoordsDataType(shape.length);
    let indicesString = '';
    if (indicesRank === 1) {
      indicesString = 'i';
    } else if (indicesRank === 2) {
      indicesString = 'i, j';
    }
    const indicesSnippet = `getIndices(${indicesString})`;

    let updatesString = '';
    if (updatesRank === 1) {
      updatesString = 'i';
    } else if (updatesRank === 2) {
      updatesString = 'i, coords[1]';
    }
    const updatesSnippet = `getUpdates(${updatesString})`;

    const strideString = sliceDim > 1 ? 'strides[j]' : 'strides';
    this.userCode = `
        ${stridesType} strides = ${stridesType}(${strides});

        void main() {
          ${dtype} coords = getOutputCoords();
          float sum = 0.0;
          bool found = false;
          for (int i = 0; i < ${updateSize}; i++) {
            int flattenedIndex = 0;
            for (int j = 0; j < ${sliceDim}; j++) {
              int index = round(${indicesSnippet});
              flattenedIndex += index * ${strideString};
            }
            if (flattenedIndex == coords[0]) {
              sum += ${updatesSnippet};
              found = true;
            }
          }
          setOutput(mix(getDefaultValue(), sum, float(found)));
        }
      `;
  }
}
