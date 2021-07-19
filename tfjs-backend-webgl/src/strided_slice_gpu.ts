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
import {getCoordsDataType} from './shader_compiler';

export class StridedSliceProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  constructor(begin: number[], strides: number[], size: number[]) {
    this.outputShape = size;
    const rank = size.length;
    const inputDtype = getCoordsDataType(size.length);
    const dtype = getCoordsDataType(size.length);

    let newCoords = '';
    if (rank === 1) {
      newCoords = 'coords * strides + begin';
    } else {
      let outputAxis = 0;
      newCoords =
          size.map((_, i) => {
                outputAxis++;
                return size.length === 1 ?
                    `coords * strides[${i}] + begin[${i}]` :
                    `coords[${outputAxis - 1}] * strides[${i}] + begin[${i}]`;
              })
              .join(',');
    }

    this.userCode = `
      ${inputDtype} begin = ${inputDtype}(${begin});
      ${inputDtype} strides = ${inputDtype}(${strides});

      void main() {
        ${dtype} coords = getOutputCoords();
        setOutput(getX(${newCoords}));
      }
    `;
  }
}
