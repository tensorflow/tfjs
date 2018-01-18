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

import * as broadcast_util from '../../broadcast_util';
import {GPGPUProgram} from './gpgpu_math';

const CHECK_NAN_SNIPPET = `
  if (isNaN(a)) return a;
  if (isNaN(b)) return b;
`;

export const ADD = 'return a + b;';
export const SUB = 'return a - b;';
export const MUL = 'return a * b;';
export const DIV = 'return a / b;';
export const POW = `
  return (round(mod(b, 2.0)) == 0 || round(mod(b, 2.0)) == 2) ?
      pow(abs(a), b) : sign(a) * pow(abs(a), b);
`;
export const EQUAL = CHECK_NAN_SNIPPET + `
  return float(a == b);
`;
export const NOT_EQUAL = CHECK_NAN_SNIPPET + `
  return float(a != b);
`;
export const LESS = CHECK_NAN_SNIPPET + `
  return float(a < b);
`;
export const LESS_EQUAL = CHECK_NAN_SNIPPET + `
  return float(a <= b);
`;
export const GREATER = CHECK_NAN_SNIPPET + `
  return float(a > b);
`;
export const GREATER_EQUAL = CHECK_NAN_SNIPPET + `
  return float(a >= b);
`;
export const LOGICAL_AND = CHECK_NAN_SNIPPET + `
  return float(a >= 1.0 && b >= 1.0);
`;
export const LOGICAL_OR = CHECK_NAN_SNIPPET + `
  return float(a >= 1.0 || b >= 1.0);
`;
export const PRELU = `
  return (a >= 0.0) ? a : b * a;
`;
export const PRELU_DER = `
  return (a > 0.0) ? 1.0 : ((a < 0.0) ? b : a);
`;
export const MAX = CHECK_NAN_SNIPPET + `
  return max(a, b);
`;
export const MIN = CHECK_NAN_SNIPPET + `
  return min(a, b);
`;

export class BinaryOpProgram implements GPGPUProgram {
  variableNames = ['A', 'B'];
  outputShape: number[];
  userCode: string;
  supportsBroadcasting = true;

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
