/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
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

export class ScatterNDProgram implements GPGPUProgram {
  variableNames = ['updates', 'indices'];
  outputShape: number[];
  userCode: string;

  constructor(
      private updateSize: number, private sliceDim: number,
      private strides: number[], shape: number[]) {
    this.outputShape = shape;
    const stridesType = getCoordsDataType(strides.length);
    const dtype = getCoordsDataType(shape.length);
    const strideString = this.sliceDim > 1 ? 'strides[j]' : 'strides';
    this.userCode = `
        ${stridesType} strides = ${stridesType}(${this.strides});

        void main() {
          ${dtype} coords = getOutputCoords();
          float sum = 0.0;
          for (int i = 0; i < ${this.updateSize}; i++) {
            int flattenIndex = 0;
            for (int j = 0; j < ${this.sliceDim}; j++) {
              int index = round(getIndices(i, j));
              flattenIndex += index * ${strideString};
            }
            if (flattenIndex == coords[0]) {
              sum += getUpdates(i, coords[1]);
            }
          }
          setOutput(sum);
        }
      `;
  }
}
