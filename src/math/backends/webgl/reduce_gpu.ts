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

import {ReduceInfo} from '../../reduce_util';
import {GPGPUProgram} from './gpgpu_math';

export class ReduceProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  constructor(reduceInfo: ReduceInfo, reduceType: 'max'|'min'|'sum') {
    const windowSize = reduceInfo.windowSize;
    const batchSize = reduceInfo.batchSize;
    const inSize = reduceInfo.inSize;
    const outSize = Math.ceil(inSize / windowSize);
    this.outputShape = [batchSize, outSize];

    const isReduceSum = reduceType === 'sum';

    let initializationValue = '0.0';
    if (!isReduceSum) {
      if (reduceType === 'min') {
        initializationValue = '1.0 / 0.0';
      } else {
        initializationValue = '-1.0 / 0.0';
      }
    }

    const compareOp = reduceType === 'min' ? 'min' : 'max';

    let returnValue = `${reduceType}(${reduceType}(${reduceType}(` +
        'minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])';
    if (reduceType === 'sum') {
      returnValue = `sumValue`;
    }

    const windowSizeNearestVec4 = Math.floor(windowSize / 4) * 4;
    const windowSizeVec4Remainder = windowSize % 4;

    const updateSnippet = `
      if (${isReduceSum}) {
        sumValue += dot(values, ones);
      } else {
        if (hasNaN(values)) {
          setOutput(getNaN(values));
          return;
        }
        minMaxValue = ${compareOp}(values, minMaxValue);
      }
    `;

    let checkOutOfBounds = '';
    if (inSize % windowSize > 0) {
      checkOutOfBounds = `
        if (inIdx < 0 || inIdx >= ${inSize}) {
          return initializationValue;
        }
      `;
    }
    this.userCode = `
      const float initializationValue = ${initializationValue};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${checkOutOfBounds}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${windowSize};

        vec4 minMaxValue = vec4(${initializationValue});
        float sumValue = 0.0;

        for (int i = 0; i < ${windowSizeNearestVec4}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${updateSnippet}
        }

        int inIdx = inOffset + ${windowSizeNearestVec4};
        if (${windowSizeVec4Remainder === 1}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );
          ${updateSnippet}
        } else if (${windowSizeVec4Remainder === 2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );
          ${updateSnippet}
        } else if (${windowSizeVec4Remainder === 3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );
          ${updateSnippet}
        }
        setOutput(${returnValue});
      }
    `;
  }
}
