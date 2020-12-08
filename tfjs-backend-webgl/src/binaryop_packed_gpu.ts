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

import {backend_util, util} from '@tensorflow/tfjs-core';

import {GPGPUProgram} from './gpgpu_math';
import {getChannels} from './packing_util';
import {getCoordsDataType} from './shader_compiler';

export const CHECK_NAN_SNIPPET = `
  result.r = isNaN.r > 0. ? NAN : result.r;
  result.g = isNaN.g > 0. ? NAN : result.g;
  result.b = isNaN.b > 0. ? NAN : result.b;
  result.a = isNaN.a > 0. ? NAN : result.a;
`;

export const ELU_DER = `
  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));
  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));
`;

export const NOT_EQUAL = `
  return vec4(notEqual(a, b));
`;

export class BinaryOpPackedProgram implements GPGPUProgram {
  variableNames = ['A', 'B'];
  outputShape: number[];
  userCode: string;
  supportsBroadcasting = true;
  packedInputs = true;
  packedOutput = true;

  constructor(
      op: string, aShape: number[], bShape: number[],
      checkOutOfBounds = false) {
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    const rank = this.outputShape.length;
    let checkOutOfBoundsString = '';
    if (checkOutOfBounds) {
      if (rank === 0 || util.sizeFromShape(this.outputShape) === 1) {
        checkOutOfBoundsString = `
          result.y = 0.;
          result.z = 0.;
          result.w = 0.;
        `;
      } else {
        const dtype = getCoordsDataType(rank);
        checkOutOfBoundsString = `
          ${dtype} coords = getOutputCoords();
        `;
        if (rank === 1) {
          checkOutOfBoundsString += `
            result.y = (coords + 1) >= ${this.outputShape[0]} ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `;
        } else {
          const channels = getChannels('coords', rank);
          checkOutOfBoundsString += `
            bool nextRowOutOfBounds =
              (${channels[rank - 2]} + 1) >= ${this.outputShape[rank - 2]};
            bool nextColOutOfBounds =
              (${channels[rank - 1]} + 1) >= ${this.outputShape[rank - 1]};
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `;
        }
      }
    }

    this.userCode = `
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${op}
      }

      void main() {
        vec4 a = getAAtOutCoords();
        vec4 b = getBAtOutCoords();

        vec4 result = binaryOperation(a, b);
        ${checkOutOfBoundsString}

        setOutput(result);
      }
    `;
  }
}
