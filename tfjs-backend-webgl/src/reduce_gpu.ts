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

import {backend_util} from '@tensorflow/tfjs-core';
import {GPGPUProgram} from './gpgpu_math';

export class ReduceProgram implements GPGPUProgram {
  variableNames = ['x'];
  outputShape: number[];
  userCode: string;

  constructor(
      reduceInfo: backend_util.ReduceInfo,
      reduceType: 'all'|'any'|'max'|'min'|'sum'|'prod') {
    const {windowSize, batchSize, inSize, outSize} = reduceInfo;
    this.outputShape = [batchSize, outSize];

    let initializationValue = '0.0';
    let compareOp = ``;

    if (reduceType === 'prod') {
      initializationValue = '1.0';
    } else if (reduceType === 'min') {
      // WebGL on Firefox Linux can't compile 1/0 so we do 1/eps.
      initializationValue = '1.0 / 1e-20';
      compareOp = `min`;
    } else if (reduceType === 'max') {
      // WebGL on Firefox Linux can't compile 1/0 so we do 1/eps.
      initializationValue = '-1.0 / 1e-20';
      compareOp = `max`;
    }

    let returnValue = `${reduceType}(${reduceType}(${reduceType}(` +
        'minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])';

    if (reduceType === 'sum') {
      returnValue = `sumValue`;
    } else if (reduceType === 'prod') {
      returnValue = `prodValue`;
    } else if (reduceType === 'all') {
      returnValue = `allValue`;
    } else if (reduceType === 'any') {
      returnValue = `anyValue`;
    }

    const windowSizeNearestVec4 = Math.floor(windowSize / 4) * 4;
    const windowSizeVec4Remainder = windowSize % 4;

    let updateSnippet = `
      if (${reduceType === 'sum'}) {
        sumValue += dot(values, ones);
      } else if (${reduceType === 'prod'}) {
        vec2 tmp = vec2(values[0], values[1]) * vec2(values[2], values[3]);
        prodValue *= tmp[0] * tmp[1];
      } else {
        minMaxValue = ${compareOp}(values, minMaxValue);
      }
    `;

    let vecType = `vec4`;

    if (reduceType === 'all') {
      initializationValue = '1.0';
      updateSnippet = `
        bool reducedAllValue = all(values);
        float floatedReducedAllValue = float(reducedAllValue);
        allValue = float(allValue >= 1.0 && floatedReducedAllValue >= 1.0);
      `;
      vecType = `bvec4`;
    } else if (reduceType === 'any') {
      initializationValue = '0.0';
      updateSnippet = `
        bool reducedAnyValue = any(values);
        float floatedReducedAnyValue = float(reducedAnyValue);
        anyValue = float(anyValue >= 1.0 || floatedReducedAnyValue >= 1.0);
      `;
      vecType = `bvec4`;
    }

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
        float prodValue = 1.0;
        float sumValue = 0.0;
        float allValue = 1.0;
        float anyValue = 0.0;

        for (int i = 0; i < ${windowSizeNearestVec4}; i += 4) {
          int inIdx = inOffset + i;
          ${vecType} values = ${vecType}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${updateSnippet}
        }

        int inIdx = inOffset + ${windowSizeNearestVec4};
        if (${windowSizeVec4Remainder === 1}) {
          ${vecType} values = ${vecType}(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          ${updateSnippet}
        } else if (${windowSizeVec4Remainder === 2}) {
          ${vecType} values = ${vecType}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          ${updateSnippet}
        } else if (${windowSizeVec4Remainder === 3}) {
          ${vecType} values = ${vecType}(
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
