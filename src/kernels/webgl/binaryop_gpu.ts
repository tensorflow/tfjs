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

import * as broadcast_util from '../../ops/broadcast_util';

import {GPGPUProgram} from './gpgpu_math';

const CHECK_NAN_SNIPPET = `
  if (isnan(a)) return a;
  if (isnan(b)) return b;
`;

export const ADD = 'return a + b;';
export const SUB = 'return a - b;';
export const MUL = 'return a * b;';

// Without the equality check div produces 0.9999 for a = b, which when
// floored can cause errors.
export const DIV = `if (a == b) return 1.0;
  return a / b;`;

// We use native integer division to deal with floating point imprecision. Since
// we implement floor division and glsl implements truncated division, we
// correct for this by subtracting 1 from result when the result is negative and
// there is a remainder.
export const INT_DIV = `
  float resultSign = sign(a) * sign(b);
  int ia = round(a);
  int ib = round(b);
  int result = ia / ib;
  int amodb = ia - ib * result;

  if (resultSign < 0.0 && amodb != 0) {
    result -= 1;
  }
  return float(result);
`;

export const POW = `
if(a < 0.0 && floor(b) < b){
  return NAN;
}
return (round(mod(b, 2.0)) != 1) ?
    pow(abs(a), b) : sign(a) * pow(abs(a), b);
`;
export const SQUARED_DIFFERENCE = 'return (a - b) * (a - b);';

export const EQUAL = `return float(a == b);`;

export const NOT_EQUAL = `return float(a != b);`;

export const LESS = `return float(a < b);`;

export const LESS_EQUAL = `return float(a <= b);`;

export const GREATER = `return float(a > b);`;

export const GREATER_EQUAL = `return float(a >= b);`;

export const LOGICAL_AND = `return float(a >= 1.0 && b >= 1.0);`;

export const LOGICAL_OR = `return float(a >= 1.0 || b >= 1.0);`;

export const MAX = CHECK_NAN_SNIPPET + `
  return max(a, b);
`;
export const MIN = CHECK_NAN_SNIPPET + `
  return min(a, b);
`;
export const MOD = `if (b == 0.0) return NAN;
  return mod(a, b);`;

export const ATAN2 = CHECK_NAN_SNIPPET + `
  return atan(a, b);
`;

export const ELU_DER = `return (b >= 1.0) ? a : a * (b + 1.0);`;

export const PRELU = `return (a < 0.) ? b * a : a;`;

export class BinaryOpProgram implements GPGPUProgram {
  variableNames = ['A', 'B'];
  outputShape: number[];
  userCode: string;

  constructor(op: string, aShape: number[], bShape: number[]) {
    this.outputShape =
        broadcast_util.assertAndGetBroadcastShape(aShape, bShape);
    this.userCode = `
      float binaryOperation(float a, float b) {
        ${op}
      }

      void main() {
        float a = getAAtOutCoords();
        float b = getBAtOutCoords();
        setOutput(binaryOperation(a, b));
      }
    `;
  }
}
