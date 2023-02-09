/**
 * @license
 * Copyright 2023 Google LLC.
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

export class ScatterPackedProgram implements GPGPUProgram {
  variableNames = ['updates', 'indices', 'defaultValue'];
  outputShape: number[];
  packedInputs = true;
  packedOutput = true;
  userCode: string;

  constructor(
      updateSize: number, sliceDim: number, indicesRank: number,
      updatesRank: number, strides: number[], shape: number[],
      summingDupeIndex = true, defaultIsTensor = false) {
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

    let defaultValuesString = '';
    if (defaultIsTensor) {
      defaultValuesString = 'coords[0], coords[1]';
    }
    const defaultValueSnippet = `getDefaultValue(${defaultValuesString})`;

    const strideString = sliceDim > 1 ? 'strides[j]' : 'strides';
    const strideString2 = sliceDim > 1 ? 'strides[j + 1]' : 'strides';

    this.userCode = `
        ${stridesType} strides = ${stridesType}(${strides});

        void main() {
          ${dtype} coords = getOutputCoords();
          vec4 sum = vec4(0.);
          vec4 found = vec4(0.);
          for (int i = 0; i < ${updateSize}; i+=2) {
            ivec2 flattenedIndex = ivec2(0);
            for (int j = 0; j < ${sliceDim}; j+=2) {
              ivec4 index = round(${indicesSnippet});
              flattenedIndex += index.xz * ${strideString};
              if (j + 1 < ${sliceDim}) {
                flattenedIndex += index.yw * ${strideString2};
              }
            }
            if (flattenedIndex[0] == coords[0] || flattenedIndex[1] == coords[0] ||
                flattenedIndex[0] == coords[0] + 1 || flattenedIndex[1] == coords[0] + 1) {
              vec4 updVals = ${updatesSnippet};
              if (flattenedIndex[0] == coords[0]) {
                sum.xy += updVals.xy;
                found.xy = vec2(1.);
              } else if (flattenedIndex[0] == coords[0] + 1) {
                sum.zw += updVals.xy;
                found.zw = vec2(1.);
              }
              if (flattenedIndex[1] == coords[0]) {
                sum.xy += updVals.zw;
                found.xy = vec2(1.);
              } else if (flattenedIndex[1] == coords[0] + 1) {
                sum.zw += updVals.zw;
                found.zw = vec2(1.);
              }
            }
          }
          setOutput(mix(${defaultValueSnippet}, sum, found));
        }
      `;
  }
}
