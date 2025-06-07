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

import {GPGPUProgram, useShapeUniforms} from './gpgpu_math';
import {getCoordsDataType, getUniformInfoFromShape, UniformType} from './shader_compiler';

export class PadProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;
  customUniforms = [{name: 'value', type: 'float' as UniformType}];
  enableShapeUniforms: boolean;

  constructor(
      xShape: number[], paddings: Array<[number, number]>,
      xTexShape: number[]) {
    this.outputShape = paddings.map(
        (p, i) => p[0] /* beforePad */ + xShape[i] + p[1] /* afterPad */);
    this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);
    const rank = xShape.length;
    const type = getCoordsDataType(rank);

    paddings.map((_, i) => {
      this.customUniforms.push({name: `pad${i}`, type: 'ivec2' as UniformType});
    });
    const start = paddings.map((_, i) => `pad${i}[0]`).join(',');
    // If the |xShape| is squeezed, we can't use it to calculate |end|.
    const {useSqueezeShape} = getUniformInfoFromShape(false, xShape, xTexShape);
    const end = paddings
                    .map(
                        (_, i) => `pad${i}[0] + ${
                            this.enableShapeUniforms && !useSqueezeShape ?
                                `xShape${rank > 1 ? `[${i}]` : ''}` :
                                xShape[i]}`)
                    .join(',');
    const unpackedCoords =
        ['coords[0]', 'coords[1]', 'coords[2]', 'coords[3]'].slice(0, rank);

    if (rank === 1) {
      this.userCode = `
        int start = ${start};
        int end = ${end};

        void main() {
          int outC = getOutputCoords();
          if (outC < start || outC >= end) {
            setOutput(value);
          } else {
            setOutput(getX(outC - start));
          }
        }
      `;
      return;
    }
    this.userCode = `
      void main() {
        ${type} start = ${type}(${start});
        ${type} end = ${type}(${end});

        ${type} outC = getOutputCoords();
        if (any(lessThan(outC, start)) || any(greaterThanEqual(outC, end))) {
          setOutput(value);
        } else {
          ${type} coords = outC - start;
          setOutput(getX(${unpackedCoords}));
        }
      }
    `;
  }
}
